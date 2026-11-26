"""Stable names shared by start-pose control and offline experiment records."""

WAIT_ONLY = 'preplan_wait_only'
PREFER_BIN = 'preplan_prefer_bin'
PREFER_WAIT = 'preplan_prefer_wait'
POLICY_BY_GROUP = {'A': WAIT_ONLY, 'B': PREFER_BIN, 'C': PREFER_WAIT}


def verified_group(summary):
    """Use effective controller settings, including archives predating named groups."""
    for group, policy in POLICY_BY_GROUP.items():
        if (summary.get('next_pick_policy') == policy
                and summary.get('direct_bin_pick') is (policy != WAIT_ONLY)
                and summary.get('next_pick_group', group) == group):
            return group
    return None
