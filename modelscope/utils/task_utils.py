from modelscope.metainfo import TaskModels
from modelscope.utils import registry
from modelscope.utils.constant import Tasks

SUB_TASKS = 'sub_tasks'
PARENT_TASK = 'parent_task'
TASK_MODEL = 'task_model'

DEFAULT_TASKS_LEVEL = {}


def _inverted_index(forward_index):
    inverted_index = dict()
    for index in forward_index:
        for item in forward_index[index][SUB_TASKS]:
            inverted_index[item] = {
                PARENT_TASK: index,
                TASK_MODEL: forward_index[index][TASK_MODEL],
            }
    return inverted_index


INVERTED_TASKS_LEVEL = _inverted_index(DEFAULT_TASKS_LEVEL)


def get_task_by_subtask_name(group_key):
    if group_key in INVERTED_TASKS_LEVEL:
        return INVERTED_TASKS_LEVEL[group_key][
            PARENT_TASK], INVERTED_TASKS_LEVEL[group_key][TASK_MODEL]
    else:
        return group_key, None
