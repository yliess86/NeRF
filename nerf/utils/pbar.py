from tqdm import tqdm as otqdm


tqdm = lambda *args, **kwargs: otqdm(*args, position=0, leave=True, **kwargs)