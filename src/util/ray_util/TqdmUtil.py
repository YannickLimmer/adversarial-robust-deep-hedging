from typing import Iterator, Generator, Any

import ray


def to_iterator(obj_ids: Iterator[ray._raylet.ObjectRef]) -> Generator[Any, None, None]:
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])


def to_limited_iterator(id_gen: Iterator[ray._raylet.ObjectRef], max_p: int):
    active_ids = [next(id_gen) for _ in range(max_p)]
    while active_ids:
        if len(active_ids) < max_p:
            try:
                active_ids.append(next(id_gen))
            except StopIteration:
                pass
        done, active_ids = ray.wait(active_ids)
        yield ray.get(done[0])
