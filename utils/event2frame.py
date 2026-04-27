import numpy as np

def event_to_cnt_img(event, height, width):
    if len(event) == 0:
        return np.zeros((2, height, width), dtype=np.float32)
    t = event[:, 0]
    p = event[:, 1].astype(np.uint8)
    x = event[:, 2].astype(np.uint32)
    y = event[:, 3].astype(np.uint32)
    # print(p)
    # print("min:", min(x))
    # print("max:", max(x))

    pos_event_x = x[p >= 1]
    pos_event_y = y[p >= 1]

    neg_event_x = x[p < 1]
    neg_event_y = y[p < 1]

    event_cnt_img_pos = np.zeros(height * width, dtype=np.float32)
    event_cnt_img_neg = np.zeros(height * width, dtype=np.float32)

    # 过滤掉超出边界的正事件
    valid_pos_mask = (pos_event_x >= 0) & (pos_event_x < width) & (pos_event_y >= 0) & (pos_event_y < height)
    valid_neg_mask = (neg_event_x < width) & (neg_event_y >= 0) & (neg_event_y < height)
    pos_event_x = pos_event_x[valid_pos_mask]
    pos_event_y = pos_event_y[valid_pos_mask]
    neg_event_x = neg_event_x[valid_neg_mask]
    neg_event_y = neg_event_y[valid_neg_mask]

    np.add.at(event_cnt_img_pos, pos_event_y * width + pos_event_x, 2)
    event_cnt_img_pos = event_cnt_img_pos.reshape([height, width])
    # print(event_cnt_img_pos.shape)

    np.add.at(event_cnt_img_neg, neg_event_y * width + neg_event_x, 2)
    event_cnt_img_neg = event_cnt_img_neg.reshape([height, width])
    # print(event_cnt_img_neg.shape)

    event_cnt_img = np.stack((event_cnt_img_pos, event_cnt_img_neg), 0)
    # print(event_cnt_img.shape)


    return event_cnt_img



def split_events_by_time(events, split_num, start_ts=None, end_ts=None):
    """
    :param events: raw events [n, 4] t, p, x, y
    :param split_num: int
    :param index_frame: index of the reconstructed image for the total results of a blurry image
    :param sum_frame: the number of the reconstructed images for a blurry image
    """
    start_ts = events[0, 0] if start_ts is None else float(start_ts)
    end_ts = events[-1, 0] if end_ts is None else float(end_ts)

    split_ts_interval = (end_ts - start_ts)/split_num

    # f_time = (end_ts - start_ts) * index_frame / (sum_frame-1) + start_ts
    # f_time = (end_ts - start_ts) * 0.5 + start_ts
    f_time = (end_ts - start_ts) * 0.5 + start_ts

    f_idx = np.searchsorted(events[:, 0], f_time)


    event_shift = events[f_idx:, :]
    event_reversal = events[:f_idx, :]

    # event polarity inverse
    event_reversal_inverse = event_reversal.copy()
    event_reversal_p = event_reversal[:, 1]
    event_reversal_inverse[:, 1] = 1-event_reversal_p

    len_shift = len(event_shift[:, 0])
    len_reversal = len(event_reversal_inverse[:, 0])

    # event shift split index
    split_shift_idx_lst = [0]
    for i in range(1, split_num + 1):
        if len_shift == 0:
            split_shift_idx_lst.append(0)
            continue
        split_ts = f_time + i * split_ts_interval
        split_idx = np.searchsorted(event_shift[:, 0], split_ts, side='right')
        #if split_idx == len_shift:
        #    split_idx = split_idx - 1
        split_shift_idx_lst.append(split_idx)

    # event reversal split index
    split_reversal_idx_lst = [len_reversal]
    for i in range(1, split_num + 1):
        if len_reversal == 0:
            split_reversal_idx_lst.append(0)
            continue
        split_ts = f_time - i * split_ts_interval
        split_idx = np.searchsorted(event_reversal_inverse[:, 0], split_ts)
        #if split_idx == 0:
        #    split_idx = split_idx - 1
        split_reversal_idx_lst.append(split_idx)

    # event shift split
    events_lst = []
    for i in range(split_num):
        start_idx = split_shift_idx_lst[i]
        end_idx = split_shift_idx_lst[i + 1]
        events_split = event_shift[start_idx:end_idx, :]
        events_lst.append(events_split)

    # event reversal split
    for i in range(split_num):
        start_idx = split_reversal_idx_lst[i + 1]
        end_idx = split_reversal_idx_lst[i]
        events_split = event_reversal_inverse[start_idx:end_idx, :]
        events_lst.append(events_split)

    return events_lst

def accumulate_event_images(event_img_list, split_num):
    total_len = len(event_img_list)
    assert total_len == 2 * split_num, "输入列表长度必须是 split_num 的两倍"

    # 分割前后两个部分
    first_half = event_img_list[:split_num]      # 前半部分
    second_half = event_img_list[split_num:]     # 后半部分

    # 前半部分倒序累积
    accumulated_first = []
    accum = np.zeros_like(first_half[0])
    for i in reversed(range(split_num)):
        accum += first_half[i]
        accumulated_first.append(accum.copy())

    # 因为是从后往前加，所以最后结果需要再 reverse 一次以保持时间顺序
    accumulated_first = accumulated_first[::-1]

    # 后半部分正序累积
    accumulated_second = []
    accum = np.zeros_like(second_half[0])
    for i in range(split_num):
        accum += second_half[i]
        accumulated_second.append(accum.copy())

    # 合并结果
    result_list = accumulated_first + accumulated_second
    return result_list