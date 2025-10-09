def dist(state1, state2):
    return

def mapper(state):
    return


def bisect_to_edge(state1, state2, abstol=1e-3):
    mapper1 = mapper(state1)
    mapper2 = mapper(state2)

    if mapper1 == mapper2:
        raise("Both initial states belong to the same basin of attraction.")
    
    d = dist(state1, state2)
    while d > abstol:
        state = (state1 + state2)/2
        if mapper(state) == mapper(state1):
            state1 = state
        elif mapper(state) == mapper(state2):
            state2 = state
        else:
            raise("Bisected state could not be mapped to any of the given attractors.")
        d = dist(state1, state2)

    return state1, state2


def edgetracking(state1, state2, eps1=1e-3, eps2=5e-3, maxiter=100):

    upper, lower, edgetrack = [], [], []

    iter = 0
    while iter <= maxiter:

        state1, state2 = bisect_to_edge(state1, state2, abstol=eps1)

        d = dist(state1, state2)
        while d < eps2:
            state1 = trajectory(state1)
            state2 = trajectort(state2)

            d = dist(state1, state2)

        upper.append(state1)
        lower.append(state2)
        edgetrack.append((state1+state2)/2)

        iter += 1
    
    return upper, lower, edgetrack