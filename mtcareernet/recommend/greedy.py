import heapq


def lazy_greedy(items, marginal, k):
    """Minoux's lazy greedy. `marginal(s, S)` is the marginal gain of `s` w.r.t. `S`."""
    heap = [(-marginal(s, set()), s, 0) for s in items]
    heapq.heapify(heap)
    S = set()
    while len(S) < k and heap:
        neg_g, s, ts = heapq.heappop(heap)
        if ts == len(S):
            S.add(s)
        else:
            new = -marginal(s, S)
            heapq.heappush(heap, (new, s, len(S)))
    return list(S)
