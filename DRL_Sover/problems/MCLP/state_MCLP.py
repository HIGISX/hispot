import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter
import math


def construct_solutions(actions):
    return actions


class StateMCLP(NamedTuple):
    # Fixed input
    users: torch.Tensor
    facilities: torch.Tensor
    p: torch.Tensor
    radius: torch.Tensor
    dist: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to   index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    # first_a: torch.Tensor
    prev_a: torch.Tensor
    visited_: torch.Tensor  # User points have been covered
    mask_cover: torch.Tensor # Facility points have been selected
    dynamic: torch.Tensor   # if one facility has been covered by another, reducing the pro to select it.
    solution: torch.Tensor  # B x p
    cover_num: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    @property
    def visited(self):
        if self.visited_.dtype == torch.bool:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))

    @staticmethod
    def initialize(data, visited_dtype=torch.bool):
        users = data["users"]
        facilities = data["facilities"]
        p = data['p'][0]
        radius = data['r'][0]
        batch_size, n_users, _ = users.size()
        _, n_facilities, _ = facilities.size()
        # dist = (facilities[:, :, None, :] - users[:, None, :, :]).norm(p=2, dim=-1)
        dist = (facilities[:, :, None, :2] - users[:, None, :, :]).norm(p=2, dim=-1)

        solution = torch.tensor([], dtype=torch.int64, device=facilities.device)
        cover_num = torch.zeros(batch_size, 1, dtype=torch.long, device=facilities.device)
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=facilities.device)

        return StateMCLP(
            users=users,
            facilities=facilities,
            p=p,
            radius=radius,
            dist=dist,
            ids=torch.arange(batch_size, dtype=torch.int64, device=facilities.device)[:, None],  # Add steps dimension
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, n_facilities,
                    dtype=torch.bool, device=users.device
                )
                if visited_dtype == torch.bool
                else torch.zeros(batch_size, 1, (n_facilities + 63) // 64, dtype=torch.int64, device=facilities.device)  # Ceil
            ),
            mask_cover=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, n_users,
                    dtype=torch.bool, device=facilities.device
                )
                if visited_dtype == torch.bool
                else torch.zeros(batch_size, 1, (n_users + 63) // 64, dtype=torch.int64, device=facilities.device)  # Ceil
            ),
            dynamic=torch.ones(batch_size, 1, n_facilities, dtype=torch.float, device=facilities.device),
            solution=solution,
            cover_num=cover_num,
            prev_a=prev_a,
            # first_a=prev_a,
            cur_coord=None,
            i=torch.zeros(1, dtype=torch.int64, device=facilities.device)  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()

        return self.cover_num

    def get_cover_num(self, facility):
        """
        :param facility: list, a list of facility index list,  if None, generate randomly
        :return: obj val of given facility_list
        """

        batch_size, n_users, _ = self.users.size()
        _, p = facility.size()
        radius = self.radius
        facility_tensor = facility.unsqueeze(-1).expand_as(torch.Tensor(batch_size, p, n_users))
        f_u_dist_tensor = self.dist.gather(1, facility_tensor)
        mask = f_u_dist_tensor < radius
        mask = torch.sum(mask, dim=1)
        cover_num = torch.count_nonzero(mask, dim=-1)

        return cover_num

    def update(self, selected):

        # Update the state
        cur_selected = selected.unsqueeze(-1)
        prev_a = cur_selected  # Add dimension for step
        cur_coord = self.facilities[self.ids, prev_a]

        cur_facility = self.solution
        new_facility = torch.cat((cur_facility, cur_selected), dim=1)
        new_cover_num = self.get_cover_num(new_facility)

        if self.visited_.dtype == torch.bool:
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a)

        # mask covered cities
        batch_size, fac_size, _ = self.facilities.size()
        # dists = (self.facilities[self.ids.squeeze(-1)] - cur_coord).norm(p=2, dim=-1).unsqueeze(1)
        dists = (self.facilities[self.ids.squeeze(-1)][..., :2] - cur_coord[..., :2]).norm(p=2, dim=-1).unsqueeze(1)
        dynamic = self.dynamic.clone()
        mask_cover = dists <= self.radius

        mask = dists > self.radius
        dynamic_update = dists.masked_fill(mask, value=1)
        dynamic = dynamic.mul(dynamic_update)

        return self._replace(prev_a=prev_a, visited_=visited_, mask_cover=mask_cover,
                             dynamic=dynamic, cover_num=new_cover_num, cur_coord=cur_coord, i=self.i + 1)

    def all_finished(self):
        # Exactly n steps
        return self.i == self.p

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        return self.visited_

    def get_dynamic(self):
        return self.dynamic
