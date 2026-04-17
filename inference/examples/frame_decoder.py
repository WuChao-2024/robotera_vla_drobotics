import traceback

import av


class H264FrameDecoder:
    """
    模拟 ROS2 DDS 订阅端：逐帧接收 msg.data 并解码 H.265。
    """
    def __init__(self):
        # 创建 H.264 解码器上下文
        self.decoder = av.codec.CodecContext.create('h264', 'r')


    def decode_one_frame(self, data: bytes):
        """
        输入一帧H.264字节流（例如DDS msg.data）
        返回解码出的图像（numpy数组）或None。
        """
        try:
            packet = av.packet.Packet(data)
            frames = self.decoder.decode(packet)
            if not frames:
                return None
            # 返回最新一帧
            frame = frames[-1]
            img = frame.to_ndarray(format='bgr24')
            return img
        except Exception as e:
            print(f"⚠️ 解码失败: {traceback.format_exc()}")
            return None
