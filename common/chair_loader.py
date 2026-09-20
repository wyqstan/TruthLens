"""安全加载 CHAIR 缓存 pkl。

``chair.pkl`` / ``chair_obj.pkl`` 是在对应类位于 ``__main__`` 时序列化的，
反序列化时 pickle 会到 ``__main__`` 查找该类。此处在加载前把类注入 ``__main__``，
避免 ``AttributeError: Can't get attribute 'CHAIR' on <module '__main__'>``。
"""
import __main__
import pickle


def load_chair(path):
    from utils.chair import CHAIR

    __main__.CHAIR = CHAIR
    try:
        from utils.chair_object365 import CHAIRObject365

        __main__.CHAIRObject365 = CHAIRObject365
    except Exception:
        pass

    with open(path, "rb") as f:
        return pickle.load(f)
