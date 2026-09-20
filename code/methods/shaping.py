"""Generic reward-shaping / observation-augmentation wrapper used by several
candidates so core.PPOLiteTrainer does not need to be forked per candidate."""
import numpy as np


class ShapedEnv:
    """Wraps an IDSPOMDPEnv, forwarding reset/step/attrs, optionally adding a reward
    bonus computed from a callback(env, action, info) -> float, and/or perturbing the
    observation adversarially before it reaches the policy."""
    def __init__(self, env, reward_bonus_fn=None, obs_perturb_fn=None):
        self._env = env
        self.reward_bonus_fn = reward_bonus_fn
        self.obs_perturb_fn = obs_perturb_fn

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, *a, **kw):
        obs = self._env.reset(*a, **kw)
        return self._maybe_perturb(obs)

    def step(self, action):
        obs, r, done, info = self._env.step(action)
        if self.reward_bonus_fn is not None:
            r += self.reward_bonus_fn(self._env, action, info)
        return self._maybe_perturb(obs), r, done, info

    def _maybe_perturb(self, obs):
        if self.obs_perturb_fn is not None:
            return self.obs_perturb_fn(self._env, obs)
        return obs


def info_gain_bonus_fn(weight=0.4):
    """Candidate 4 (Budgeted Active Subgraph Acquisition): rewards INFO actions in
    proportion to how much they change the agent-visible (masked-feature) belief,
    using the env's own probe scored on the *masked* vector as a cheap onboard belief
    model -- never the oracle label."""
    from env.pomdp_env import INFO_ACTIONS
    prev_belief = {}

    def bonus(env, action, info):
        i_prev = env.queue[0] if env.queue else None
        if action not in INFO_ACTIONS or i_prev is None:
            return 0.0
        mask = np.zeros(env.n_features, dtype=np.float32)
        for g in env.acquired_groups.get(i_prev, []):
            lo, hi = env.group_bounds[g], env.group_bounds[g + 1]
            mask[lo:hi] = 1.0
        masked_x = (env.X[i_prev] * mask).reshape(1, -1)
        try:
            belief = float(env.probe.predict_proba(masked_x)[0, 1])
        except Exception:
            belief = 0.5
        prev = prev_belief.get(i_prev, 0.5)
        gain = abs(belief - prev)
        prev_belief[i_prev] = belief
        return weight * gain
    return bonus
