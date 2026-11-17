import math


def cosine_scheduler(start, end, max_steps, current_step):
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * current_step / max_steps))


def linear_scheduler(start, end, max_steps, current_step):
    return start + (end - start) * (current_step / max_steps)


def linear_delta_scheduler(start, end, max_steps, current_step, delta):
    return start + (end - start) * (current_step / max_steps)


class PruneScheduler:
    def __init__(self, sched_type, start, end, max_steps, delta):
        if sched_type == "cosine":
            self.hard_prune_sched = cosine_scheduler
        elif sched_type == "linear":
            self.hard_prune_sched = linear_scheduler
        else:
            raise ValueError(f"Invalid scheduler: {sched_type}")

        self.start = start
        self.end = end
        self.max_steps = max_steps
        self.delta = delta
        assert delta >= 0 and delta <= 1
        self.end_step = self.max_steps * self.delta

    def get_prune_kept_ratio(self, current_step):
        if current_step >= self.end_step:
            current_step = 0
        return self.hard_prune_sched(
            start=self.start, end=self.end, max_steps=self.end_step, current_step=current_step
        )
