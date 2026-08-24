# [fused_opt版] = serve_m7_eval_bpu_fused.py 优化版(2026-08-19), 优化手法照搬 gr00t_release/serve_m7_eval_bpu_opt.py。
# 数值语义与 fused 版一致(同款对拍验证), 所有改动点均带 [opt] 标注。三板斧:
# 1) vision 3 摄: 单实例 n_task=3, 顺序 图像前处理+IONArray直写+start(task_id=i), 统一 wait_done
#    (BPU 3 核并行; 3 线程并行实测反噬 GIL+cv2 过订阅, 弃用——opt 版结论)
# 2) zero-copy: vision/lang/act 输入 IONArray 常驻绑定跨帧复用; lang 输出 vl IONArray 直绑 act 输入[3]
#    (同物理内存, Euler 4 步共享); act 输出(下一步 actions)直绑回 act 输入[1](fused 特有, Euler 回喂免拷贝)
# 3) CPU 管线重排: text 预处理按 text 缓存(bp 只依赖 text, images 只定恒定 grid_thw) + 提前到 BPU 空闲窗口;
#    vision BPU 窗口内预计算 iam/niam/ie/packed; state57/iam/niam 每帧只写一次。
"""板端 BPU 推理 serve·fused 优化版。ZMQ REP 协议与 serve_m7_eval_bpu_fused.py 完全一致:
recv np.savez_compressed(images=dict, state[N,57], text) -> send_pyobj({"actions":[N,16,38]})
run: 同 fused 版参数(无 --feed)。
"""
import argparse
import io
import json
import os
import threading
import time

os.environ.setdefault("GR00T_M7_URDF", os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpu_model", "v1", "l3_4.urdf"))

import cv2
import numpy as np
import zmq
from camera_frame import cam_absolute, fk_base_rs
from pyCauchyKesai import CauchyKesai, IONArray

MASK_NEG = -32767.0
STATE_DIM = 57
AH = 16
ACT_HORIZON = 40
ACTION_DIM = 132
NUM_STEPS = 4
ACTION_KEYS = ["right_end_pose", "right_hand", "left_end_pose", "left_hand"]
STATE_SLICES = [("right_arm_joint", 0, 7), ("left_arm_joint", 7, 14), ("right_end_pose", 14, 21),
                ("left_end_pose", 21, 28), ("right_hand", 28, 40), ("left_hand", 40, 52),
                ("waist", 52, 55), ("neck", 55, 57)]
ACTION_SLICES = [("right_end_pose", 0, 7), ("right_hand", 7, 19), ("left_end_pose", 19, 26), ("left_hand", 26, 38)]
EVAL_CAM_KEYS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

_T = {}  # [opt] 分段计时
def _mark(name, t0):
    _T[name] = _T.get(name, 0.0) + (time.time() - t0) * 1000.0


def _smallest_max_size(img, m=256):
    H, W = img.shape[:2]
    s = m / float(min(H, W))
    return cv2.resize(img, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_AREA)


# [opt·LUT 2026-08-19, 板端 bench 逐位一致] uint8 只有 256 种取值——normalize 按原版公式
# ((i/255-0.5)/0.5, fp32 序) 离线枚举成 fp16 查找表, uint8->fp16 一遍出, 跳过整条 fp32 链与
# 慢速 astype(fp16)(实测 1.65ms/摄)。三摄 16.5->10.9ms 且逐位一致。
_LUT = ((np.arange(256, dtype=np.float32) * (1.0 / 255.0) - 0.5) / 0.5).astype(np.float16)


def img_to_pixel(img_uint8_hwc):
    # [改动记录 2026-08-19] normalize/patchify 段改 LUT 实现(逐位等价, 板端 bench 验证);
    # resize 几何与原版逐字节同函数。原版实现:
    #   x = img.astype(np.float32) * (1.0/255.0); x = (x - 0.5) / 0.5; x = x.transpose(2, 0, 1)
    #   patches = np.stack([x, x], axis=0)
    #   y = patches.reshape(1,2,3,8,2,16,8,2,16).transpose(0,3,6,4,7,2,1,5,8)
    #   return y.reshape(256,1536).astype(np.float16)
    img = _smallest_max_size(img_uint8_hwc, 256)
    H, W = img.shape[:2]
    ch, cw = int(H * 0.95), int(W * 0.95)
    t, l = (H - ch) // 2, (W - cw) // 2
    img = img[t:t + ch, l:l + cw]
    img = _smallest_max_size(img, 256)
    H, W = img.shape[:2]
    t, l = max(0, (H - 256) // 2), max(0, (W - 256) // 2)
    img = img[t:t + 256, l:l + 256]
    x16 = _LUT[img]                                     # [256,256,3] fp16 一遍(逐位=原版)
    x16 = np.ascontiguousarray(x16.transpose(2, 0, 1))  # [3,256,256] fp16
    p = np.empty((2, 3, 256, 256), dtype=np.float16)
    p[0] = x16; p[1] = x16                              # tp=2 单帧复制
    y = p.reshape(1, 2, 3, 8, 2, 16, 8, 2, 16).transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return y.reshape(256, 1536)


def unnorm_action(action_pred, raw_state57, stats):
    # (与 fused 版逐字节同函数, 含 quat 规范化)
    raw = {k: raw_state57[s:e].astype(np.float64) for k, s, e in STATE_SLICES}
    T_base_rs = fk_base_rs(raw["waist"], raw["neck"])
    out = {}
    for k, a, b in ACTION_SLICES:
        chunk = action_pred[0, :AH, a:b].astype(np.float64)
        if k in ("right_end_pose", "left_end_pose"):
            pmin = np.array(stats["relative_action"][k]["min"], dtype=np.float64)
            pmax = np.array(stats["relative_action"][k]["max"], dtype=np.float64)
        else:
            pmin = np.array(stats["action"][k]["q01"], dtype=np.float64)
            pmax = np.array(stats["action"][k]["q99"], dtype=np.float64)
        unn = (np.clip(chunk, -1, 1) + 1) / 2 * (pmax - pmin) + pmin
        if k in ("right_end_pose", "left_end_pose"):
            qp = cam_absolute(unn, raw[k], T_base_rs)
            w = qp[:, 6:7]
            qp[:, 3:7] = np.where(w < 0, -qp[:, 3:7], qp[:, 3:7])
            out[k] = qp
        else:
            out[k] = unn
    act38 = np.concatenate([out[k] for k in ACTION_KEYS], -1)
    return act38[None].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vbm", required=True)
    ap.add_argument("--lbm", required=True)
    ap.add_argument("--abm", required=True, help="action_fused.hbm")
    ap.add_argument("--stats", required=True)
    ap.add_argument("--cosmos", required=True)
    ap.add_argument("--embed", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5557)
    args = ap.parse_args()
    cv2.setNumThreads(8)  # [opt·3线程] 8 核板全开(用户指定); 3 前处理线程 x cv2 内部并行

    # [opt] vision: 单实例 n_task=3, 3 slot 输入 IONArray 常驻绑定
    vck = CauchyKesai(args.vbm, n_task=3)
    v_ions = [IONArray(vck.input_descs[0]) for _ in range(3)]
    for i in range(3):
        assert vck.check_input(v_ions[i], 0), f"vision ion slot{i} check fail"
        vck.inputs[i][0] = v_ions[i]
    lck = CauchyKesai(args.lbm)
    ack = CauchyKesai(args.abm)
    # [opt] lang/act 输入 IONArray 常驻绑定
    l_ions = [IONArray(d) for d in lck.input_descs]
    for i, ion in enumerate(l_ions):
        assert lck.check_input(ion, i), f"lang ion {i} check fail"
        lck.inputs[0][i] = ion
    a_ions = [IONArray(d) for d in ack.input_descs]
    for i, ion in enumerate(a_ions):
        assert ack.check_input(ion, i), f"act ion {i} check fail"
        ack.inputs[0][i] = ion

    from board_preprocess import BoardPreprocess
    _bp_raw = BoardPreprocess(args.cosmos, args.embed, ["cam_high", "cam_left", "cam_right"])
    # [opt v4] bp() 输出只依赖 text(见 gr00t_release opt 版论证), 按 text 缓存逐位复现
    _bp_cache = {}

    def bp(text, images_bp):
        r = _bp_cache.get(text)
        if r is None:
            if len(_bp_cache) >= 64:
                _bp_cache.clear()
            r = _bp_raw(text, images_bp)
            _bp_cache[text] = r
        return r
    print(f"[serve_bpu_fused_opt] BoardPreprocess ready (cosmos={args.cosmos})", flush=True)

    stats = json.load(open(args.stats))["new_embodiment"]

    # [opt·3线程 2026-08-19] 三摄前处理真并行(cv2.setNumThreads(2) 防过订阅——旧「线程反噬」
    # 结论的真凶是 cv2 默认 18 线程过订阅; numpy/cv2 的 C 级操作释放 GIL)。板端 bench:
    # 三摄前处理 16.5 -> 4.5ms 墙钟。ion 写/start 留主线程(保守, 避开 pyCauchyKesai 线程安全未知)。
    _px_buf = [None] * 3

    def _prep_thread(idx, img):
        _px_buf[idx] = img_to_pixel(img)

    def vision_stage(cam_imgs_):
        t0 = time.time()
        ths = [threading.Thread(target=_prep_thread, args=(ci, cam_imgs_[ek]))
               for ci, ek in enumerate(EVAL_CAM_KEYS)]
        [th.start() for th in ths]; [th.join() for th in ths]
        _mark("img_prep", t0)  # 三摄并行墙钟(原顺序版为三摄串行总和)
        t0 = time.time()
        for i in range(3):
            v_ions[i].numpy()[:] = _px_buf[i]
            v_ions[i].flush_clean()
            vck.start(task_id=i)
        _mark("vision_submit", t0)

    ctx = zmq.Context(); sock = ctx.socket(zmq.REP); sock.bind(f"tcp://{args.host}:{args.port}")
    print(f"[serve_bpu_fused_opt] listening tcp://{args.host}:{args.port}", flush=True)

    # [opt1] scatter 单写化的跨帧状态(上次全量直写时的 text; 命中则模板位沿用常驻 IONArray)
    _shadow_text = [None]
    _shadow_ready = [False]

    lat_hist = []
    n_query = 0
    while True:
        msg = sock.recv()
        t_q0 = None  # [口径修正 2026-08-19] 推理延迟按 C 口径: parse 完成(数据在内存可用)起算——
        # 压缩报文的 zlib 解压/pickle 解析属传输协议解码, 不属推理(parse 随评测端压缩选择而变)
        try:
            _T.clear(); _q0 = time.time()
            data = np.load(io.BytesIO(msg), allow_pickle=True)
            files = data.files
            images_raw = data["images"].item() if "images" in files else {}
            state_raw = np.asarray(data["state"], dtype=np.float32)
            text_raw = data["text"] if "text" in files else ""
            if state_raw.ndim == 1:
                state_raw = state_raw[None]
            N = state_raw.shape[0]
            if state_raw.shape[1] != STATE_DIM:
                raise ValueError(f"state dim expect {STATE_DIM}, got {state_raw.shape[1]}")
            for ek in EVAL_CAM_KEYS:
                if ek not in images_raw:
                    raise ValueError(f"missing image key '{ek}'")

            _t = text_raw[0] if isinstance(text_raw, np.ndarray) and text_raw.ndim > 0 and text_raw.size else str(text_raw)
            text_str = str(_t)
            _mark("parse", _q0)
            t_q0 = time.time()  # [口径修正] C 口径起点: parse 完成

            actions_all = []
            for ni in range(N):
                st57 = state_raw[ni]
                cam_imgs = {}
                for ek in EVAL_CAM_KEYS:
                    arr = np.asarray(images_raw[ek])
                    if arr.dtype != np.uint8:
                        arr = arr.astype(np.uint8)
                    if arr.ndim == 3:
                        arr = arr[None]
                    cam_imgs[ek] = arr[ni]

                # [opt v3] CPU 先行: text_prep 在无 BPU 争抢时做
                images_bp = {"cam_high": [cam_imgs["cam_high"]],
                             "cam_left": [cam_imgs["cam_left_wrist"]],
                             "cam_right": [cam_imgs["cam_right_wrist"]]}
                t0 = time.time()
                bo = bp(text_str, images_bp)
                _mark("text_prep", t0)
                ip = bo["img_pos"][0]; seq = bo["input_embeds"].shape[1]

                # [opt] 3 摄前处理并行 + 顺序零拷贝写/异步 start(3 task 在 BPU 3 核并行)
                vision_stage(cam_imgs)

                # [opt v3] vision BPU 窗口内预计算 CPU(mask/ie/packed zeros; fused 无 state 编码段)
                t0 = time.time()
                bm = (bo["attn_mask"][0, :seq] == 1)
                im = ip[:seq]
                ik = (im & bm).reshape(1, 1, 1, seq)
                nk = ((~im) & bm).reshape(1, 1, 1, seq)
                iam = np.broadcast_to(np.where(ik, 0.0, MASK_NEG), (1, 1, 41, seq)).astype(np.float16)
                niam = np.broadcast_to(np.where(nk, 0.0, MASK_NEG), (1, 1, 41, seq)).astype(np.float16)
                state_in16 = st57.astype(np.float32).reshape(1, 1, 57).astype(np.float16)
                # [opt1] text 命中时 ie/packed 无需构造(常驻 IONArray 视图直接 scatter); 变化才构造
                if not (_shadow_text[0] == text_str and _shadow_ready[0]):
                    ie = bo["input_embeds"].copy()
                    packed = [np.zeros((1, seq, 2048), dtype=np.float16) for _ in range(3)]
                _mark("pre_cpu", t0)

                # [opt] 统一等 3 摄结束, 输出 IONArray 视图直读(零拷贝)
                t0 = time.time()
                for i in range(3):
                    vck.wait_done(task_id=i)
                _mark("vision_bpu", t0)
                # [opt 2026-08-19] vision 输出保持 fp16 直 scatter(原版 fp32 中转再 astype(fp16) 是
                # fp16->fp32->fp16 无损往返, 删掉中转省 4 次 astype; concat 不运算与 dtype 无关)
                img_embs = []; ds_all = [[] for _ in range(3)]
                for i in range(3):
                    outs = vck.outputs[i]
                    img_embs.append(np.asarray(outs[0].numpy()))
                    for k in range(3):
                        ds_all[k].append(np.asarray(outs[1 + k].numpy()))
                image_embeds = np.concatenate(img_embs, axis=1)      # fp16
                deepstack = [np.concatenate(ds_all[k], axis=1) for k in range(3)]  # fp16

                t0 = time.time()
                # [opt1·正式 2026-08-19, debug 版 shadow 双算验证逐位 0] text 命中时模板位/零位
                # 与上帧完全相同——直接在常驻 IONArray 视图上只 scatter image 位, 跳过
                # 「模板整拷贝 + 构造零张量 + 4.2MB 整块直写」; text 变化才全量直写(正确性回退路径)。
                # position_ids/attention_mask/pad_mask(小输入)不变, 仍整写。
                if _shadow_text[0] == text_str and _shadow_ready[0]:
                    l0_view = l_ions[0].numpy()
                    l0_view[0, ip] = image_embeds.reshape(-1, 2048)
                    l_ions[0].flush_clean()
                    for k in range(3):
                        pk_view = l_ions[4 + k].numpy()
                        pk_view[0, ip] = deepstack[k].reshape(-1, 2048)
                        l_ions[4 + k].flush_clean()
                else:
                    ie[0, ip] = image_embeds.reshape(-1, 2048)
                    for k in range(3):
                        packed[k][0, ip] = deepstack[k].reshape(-1, 2048)
                    for i, arr in enumerate([ie, bo["position_ids"], bo["attention_mask"], bo["pad_mask"],
                                             packed[0], packed[1], packed[2]]):
                        l_ions[i].numpy()[:] = arr
                        l_ions[i].flush_clean()
                    _shadow_text[0] = text_str; _shadow_ready[0] = True
                    _mark("scatter_full", t0)
                # 3 个小输入(position_ids/attention_mask/pad_mask, 共 ~130KB)每帧直写不变量仍整写
                for i, arr in enumerate([bo["position_ids"], bo["attention_mask"], bo["pad_mask"]]):
                    l_ions[1 + i].numpy()[:] = arr
                    l_ions[1 + i].flush_clean()
                lck.start(task_id=0)
                _mark("scatter", t0)

                t0 = time.time()
                lck.wait_done(task_id=0)
                vl_ion = lck.outputs[0][0]             # [opt] lang 输出 IONArray(零拷贝)
                _mark("lang_bpu", t0)

                # [opt] lang 输出直绑 act 输入[3](fused 签名 vl 是 index 3; 同物理内存, Euler 4 步共享)
                vl_zero_copy = ack.check_input(vl_ion, 3)
                if vl_zero_copy:
                    ack.inputs[0][3] = vl_ion
                else:
                    a_ions[3].numpy()[:] = np.asarray(vl_ion.numpy()).astype(np.float16)
                    a_ions[3].flush_clean()

                # [opt] 每帧不变量只写一次: state57(输入0)/iam(4)/niam(5)
                t0 = time.time()
                a_ions[0].numpy()[:] = state_in16
                a_ions[0].flush_clean()
                a_ions[4].numpy()[:] = iam
                a_ions[4].flush_clean()
                a_ions[5].numpy()[:] = niam
                a_ions[5].flush_clean()
                _mark("act_static_in", t0)

                noise = (data["noise"].astype(np.float32) if "noise" in files
                         else np.random.randn(1, ACT_HORIZON, ACTION_DIM).astype(np.float32))
                # [opt fused 特有] Euler 回喂零拷贝尝试: 图输出(下一步 actions, fp16)直绑回输入[1]。
                # fp16->fp32->fp16 无损, 数值与 fused 版的显式 cast 回喂逐位一致。
                # 注意: 上帧若直绑过, inputs[0][1] 还指着旧 out_ion——帧首恢复常驻绑定再写 noise。
                t0 = time.time()
                ack.inputs[0][1] = a_ions[1]
                a_ions[1].numpy()[:] = noise.astype(np.float16)
                a_ions[1].flush_clean()
                actions_feed_zero_copy = False
                for t in range(NUM_STEPS):
                    t0b = time.time()
                    a_ions[2].numpy()[:] = np.array([t], dtype=np.int32)
                    a_ions[2].flush_clean()
                    _mark("act_step_in", t0b)
                    t0h = time.time()
                    ack.start(task_id=0)
                    ack.wait_done(task_id=0)
                    out_ion = ack.outputs[0][0]
                    _mark("act_bpu", t0h)
                    if t < NUM_STEPS - 1:
                        # 下一步 actions 回喂: 优先输出直绑输入, 不行则显式 copy(fp16)
                        if ack.check_input(out_ion, 1):
                            ack.inputs[0][1] = out_ion
                            actions_feed_zero_copy = True
                        else:
                            a_ions[1].numpy()[:] = np.asarray(out_ion.numpy()).astype(np.float16)
                            a_ions[1].flush_clean()
                actions = np.asarray(out_ion.numpy()).astype(np.float32)  # [1,40,132] 最终步
                _mark("euler_all", t0)
                t0 = time.time()
                actions_all.append(unnorm_action(actions, st57, stats)[0])
                _mark("post_unnorm", t0)
            actions_arr = np.stack(actions_all, axis=0).astype(np.float32)
            n_query += 1
            lat_ms = (time.time() - t_q0) * 1000
            lat_hist.append(lat_ms)
            a0 = actions_arr[0, 0]
            _parts = " ".join(f"{k}={v:.1f}" for k, v in _T.items())
            print(f"[serve_bpu_fused_opt] q#{n_query} t={lat_ms:.0f}ms(ema {sum(lat_hist[-50:])/len(lat_hist[-50:]):.0f}) "
                  f"text={text_raw!r} N={N} vzc={int(vl_zero_copy)} azc={int(actions_feed_zero_copy)}", flush=True)
            print(f"\033[31m[perf] q#{n_query} total={(time.time()-_q0)*1000:.1f}ms | {_parts}\033[0m", flush=True)
            sock.send_pyobj({"actions": actions_arr})
        except Exception as e:
            import traceback; traceback.print_exc()
            sock.send_pyobj({"error": f"serve_bpu_fused_opt: {e}"})


if __name__ == "__main__":
    main()
