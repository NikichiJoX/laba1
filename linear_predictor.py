import numpy as np
import learned_context as lc


def spatial_features(padded, y, x):
    left = padded[y, x - 1]
    above = padded[y - 1, x]
    above_left = padded[y - 1, x - 1]
    above_right = padded[y - 1, x + 1]

    last_column = padded.shape[1] - 2
    flags = np.stack([np.ones_like(x), x == 1, y == 1, x == last_column], axis=-1)
    neighbors = [flags, left, above, above_left, above_right]
    features = np.concatenate(neighbors, axis=-1)

    difference = left.astype(np.int32) - above
    activity = np.abs(difference).sum(axis=-1)
    return features, activity


def fit_predictor(q, b):
    n, height, width, channels = q.shape
    alphabet_size = 2 ** b
    y, x = np.indices((height, width)) + 1
    feature_blocks = []
    activity_blocks = []

    for image in q:
        padded = np.pad(image, ((1, 1), (1, 1), (0, 0)))
        neighbors, activity = spatial_features(padded, y, x)
        block = np.concatenate([neighbors, image], axis=-1)
        feature_blocks.append(block)
        activity_blocks.append(activity)

    features = np.stack(feature_blocks)
    design = features.reshape(n * height * width, -1).astype(np.float64)
    target = q.reshape(-1, channels)
    feature_count = design.shape[1]
    gram = design.T @ design
    gram += 0.1 * np.eye(feature_count)
    cross = design.T @ target

    neighbor_count = feature_count - channels
    weights = np.zeros((feature_count, channels), dtype=np.int32)
    for c in range(channels):
        end = neighbor_count + c
        matrix = gram[:end, :end]
        column = cross[:end, c]
        coefficients = np.linalg.solve(matrix, column)
        weights[:end, c] = np.rint(256 * coefficients).astype(np.int32)

    scores = design.astype(np.int32) @ weights
    bin_count = 4 * alphabet_size
    bins = (scores + 32) // 64
    bins = np.clip(bins, 0, bin_count - 1)

    activity = np.asarray(activity_blocks).ravel()
    quartiles = np.quantile(activity, [0.25, 0.5, 0.75])
    limits = quartiles.astype(np.int32)
    bands = np.searchsorted(limits, activity)
    counts = np.zeros((channels, bin_count, 4, alphabet_size))
    for c in range(channels):
        indices = (bins[:, c], bands, target[:, c])
        np.add.at(counts[c], indices, 1)

    prior = counts.sum(axis=2, keepdims=True) + 0.5
    prior /= prior.sum(axis=-1, keepdims=True)
    probabilities = lc.smooth(counts, prior, 12)
    table = lc.frequencies(probabilities)
    return weights, limits, table


def prediction_rows(padded, model):
    weights, limits, table = model
    height, width, channels = padded.shape
    neighbor_weights = weights[:-channels]
    channel_weights = weights[-channels:]
    last_bin = table.shape[1] - 1

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            features, activity = spatial_features(padded, y, x)
            scores = features @ neighbor_weights
            band = np.searchsorted(limits, activity)
            for c in range(channels):
                index = (scores[c] + 32) // 64
                index = min(max(index, 0), last_bin)
                yield (y, x, c), table[c, index, band]

                symbol = int(padded[y, x, c])
                scores += symbol * channel_weights[c]


def encode_linear(path, q, scale, b, model):
    padded = np.pad(q, ((1, 1), (1, 1), (0, 0)))
    alphabet_size = 2 ** b
    header = lc.HEADER.pack(b"NIC1", 3, b, *q.shape, scale)
    with path.open("wb") as stream:
        stream.write(header)
        coder = lc.ArithmeticEncoder(16, stream)
        for position, frequencies in prediction_rows(padded, model):
            symbol = int(padded[position])
            coder.write(frequencies, symbol, alphabet_size)
        coder.finish()


def decode_linear(path, model):
    with path.open("rb") as stream:
        header = stream.read(lc.HEADER.size)
        magic, mode, b, height, width, channels, scale = lc.HEADER.unpack(header)
        if magic != b"NIC1" or mode != 3:
            raise ValueError("Ожидался поток линейного предиктора")
        padded = np.zeros((height + 2, width + 2, channels), dtype=np.uint8)
        alphabet_size = 2 ** b
        coder = lc.ArithmeticDecoder(16, stream)
        for position, frequencies in prediction_rows(padded, model):
            cumulative = np.r_[0, np.cumsum(frequencies[:-1])]
            symbol = coder.read(frequencies, cumulative, alphabet_size)
            padded[position] = symbol

    q = padded[1:-1, 1:-1]
    return lc.dequantize(q, scale, b)
