import numpy as np


class SA:
    def __init__(self, table_count, guest_list, relationships_mat,
                 temp=1.0, temp_min=0.00000001, alpha=0.9, n_iter=200, audit=True):
        self.table_count = table_count
        self.guest_list = guest_list
        self.relationships_mat = relationships_mat
        self.temp = temp
        self.temp_min = temp_min
        self.alpha = alpha
        self.n_iter = n_iter
        self.audit = audit

    # initialize a seating randomly
    def init_pos(self):
        s = (self.table_count, len(self.guest_list))
        pos_current = np.zeros(s)
        for i in range(self.table_count):
            pos_current[i][10*(i):min(10*(i+1),len(self.guest_list))] = 1
        return pos_current

    # make sure the matrix is of the right shape
    def reshape_to_table_seats(self, x):
        table_seats = x.reshape(self.table_count, len(self.guest_list))
        return table_seats

    # objective function
    def cost(self, x):
        table_seats = self.reshape_to_table_seats(x)
        # actual cost
        table_costs = table_seats * self.relationships_mat * table_seats.T
        table_cost = np.trace(table_costs)
        # penalty if some table has more than the the table can host
        if sum(i > 10 for i in table_seats.sum(axis=1)) > 0:
            penalty = 1e10
        else:
            penalty = 0
        return table_cost + penalty

    # randomly swap two seats (with and without guest)
    def take_step(self, table_seats):
        table_seats = self.reshape_to_table_seats(np.matrix(table_seats, copy=True))
        # randomly swap one guest with another seat (can be empty)

        table_from, table_to = np.random.choice(self.table_count, 2, replace=False)

        table_from_guests = np.where(table_seats[table_from] == 1)[1]

        # make sure table from is a guest
        while len(table_from_guests) == 0:
            table_from, table_to = np.random.choice(self.table_count, 2, replace=False)
            table_from_guests = np.where(table_seats[table_from] == 1)[1]
        # all guests at the table to be exchanged to
        table_to_guests = np.where(table_seats[table_to] == 1)[1]
        # original guest
        table_from_guest = np.random.choice(table_from_guests)
        # randomly choose one
        table_to_guest = np.random.choice(table_to_guests)

        # update seating charts
        table_seats[table_from, table_from_guest] = 0
        table_seats[table_from, table_to_guest] = table_seats[table_to, table_to_guest]
        table_seats[table_to, table_to_guest] = 0
        table_seats[table_to, table_from_guest] = 1
        return table_seats

    # metropolis criterion
    @staticmethod
    def prob_accept(cost_old, cost_new, temp):
        a = 1 if cost_new < cost_old else np.exp((cost_old - cost_new) / temp)
        return a

    # annealing
    def anneal(self):

        # initialization
        pos_current = self.init_pos()

        # initial cost
        cost_old = self.cost(pos_current)

        audit_trail = []

        while self.temp > self.temp_min:
            for i in range(0, self.n_iter):
                pos_new = self.take_step(pos_current)
                cost_new = self.cost(pos_new)
                p_accept = self.prob_accept(cost_old, cost_new, self.temp)
                if p_accept > np.random.random():
                    pos_current = pos_new
                    cost_old = cost_new
                if self.audit:
                    audit_trail.append((cost_new, cost_old, self.temp, p_accept))
            self.temp *= self.alpha

        return pos_current, cost_old, audit_trail
