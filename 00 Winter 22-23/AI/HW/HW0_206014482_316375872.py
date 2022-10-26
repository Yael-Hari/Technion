from typing import List
import numpy as np


# -------------- Q1 --------------


def twoSum(nums: List[int], target: int) -> List[int]:
    # assumption: exactly 1 solution;

     # sort list: O(nlogn)
    mergeSort(nums, 0, len(nums) - 1)

    # find: O(n)
    start = 0
    end = len(nums) - 1

    sum = nums[start] + nums[end]

    while sum != target or start == end:
        if sum > target:
            end = end - 1
        if sum < target:
            start = start + 1

        sum = nums[start] + nums[end]

    return [start, end]


def merge(arr, l, m, r):
    n1 = m - l + 1
    n2 = r - m

    # create temp arrays
    L = [0] * (n1)
    R = [0] * (n2)

    # Copy data to temp arrays L[] and R[]
    for i in range(0, n1):
        L[i] = arr[l + i]

    for j in range(0, n2):
        R[j] = arr[m + 1 + j]

    # Merge the temp arrays back into arr[l..r]
    i = 0  # Initial index of first subarray
    j = 0  # Initial index of second subarray
    k = l  # Initial index of merged subarray

    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1

    # Copy the remaining elements of L[], if there
    # are any
    while i < n1:
        arr[k] = L[i]
        i += 1
        k += 1

    # Copy the remaining elements of R[], if there
    # are any
    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1


def mergeSort(arr, l, r):
    if l < r:
        # Same as (l+r)//2, but avoids overflow for
        # large l and h
        m = l + (r - l) // 2

        # Sort first and second halves
        mergeSort(arr, l, m)
        mergeSort(arr, m + 1, r)
        merge(arr, l, m, r)



# -------------- Q2 --------------

def optimalProfit(prices):
    # we want: max { prices[j]  - prices[i] } s.t. i<j
    max_profit = 0
    for sell in range(len(prices)):
        for buy in range(sell):
            curr_profit = prices[sell] - prices[buy]
            if curr_profit > max_profit:
                max_profit = curr_profit

    return max_profit


# -------------- Q3 --------------


class Node:

    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node

    def __str__(self):
        return str(self.value)


# -------------- Q3.1 --------------


def read_file(file_path: str) -> Node:
    with open(file_path, 'r') as f:
        text = f.readline()

    nodes_vals_list = text.split(',')
    nodes_vals_list = [int(x) for x in nodes_vals_list]

    nodes_vals_list.reverse()
    nodes_vals_list_reversed = nodes_vals_list

    tail = Node(value=nodes_vals_list_reversed[0])
    next_node = tail

    for val in nodes_vals_list_reversed[1:]:
        next_node = Node(value=val, next_node=next_node)

    head = next_node
    return head


# -------------- Q3.2 --------------


def get_length(head: Node) -> int:
    length = 0
    next_node = head
    while next_node is not None:
        length += 1
        next_node = next_node.next
    return length


# -------------- Q3.3 --------------

def sort_in_place(head: Node) -> Node:
    # bubble sort
    switch_flag = True
    new_head = Node(value=-np.inf, next_node=head)

    while switch_flag:
        switch_flag = False

        before_node = new_head
        curr_node = before_node.next
        while curr_node.next is not None:
            if curr_node.value == 2:
                print()
            curr_node = before_node.next
            next_node = curr_node.next
            if curr_node.value > next_node.value:
                swap_nodes(before_node, curr_node, next_node)
                switch_flag = True
                before_node = next_node
                curr_node = before_node.next

            else:
                before_node = curr_node
                curr_node = before_node.next

    return new_head.next


def swap_nodes(before_node, curr_node, next_node):
    before_node.next = next_node
    curr_node.next = next_node.next
    next_node.next = curr_node


if __name__ == '__main__':
    # print(twoSum([1, 3, 5, 6, 7, 11], 9))
    # print(optimalProfit([7, 1, 5, 3, 6, 4]))
    head = read_file('test.txt')
    sort_in_place(head)

