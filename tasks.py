tasks_vis={
    "task1":{0: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,
             20,21,22,23,24,25,26,27,28,29],
             1:[30,31,32,33,34],
             2:[35,36,37,38,39]
             },
    "task2":{0: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
             1:[20,21,22,23,24],
             2:[25,26,27,28,29],
             3:[30,31,32,33,34],
             4:[35,36,37,38,39]
             },
    'task3':{0: [0,1,2,3],
             1: [4,5,6,7],
             2:[8,9,10,11],
             3:[12,13,14,15],
             4:[16,17,18,19],
             5:[20,21,22,23],
             6:[24,25,26,27],
             7:[28,29,30,31],
             8:[32,33,34,35],
             9:[36,37,38,39]
             },
    "task4":{0: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
             1:[20,21,22,23],
             2:[24,25,26,27],
             3:[28,29,30,31],
             4:[32,33,34,35],
             5:[36,37,38,39],
            },
    "20_2":{0: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
        1: [20,21],
        2: [22,23],
        3: [24,25],
        4: [26,27],
        5: [28,29],
        6: [30,31],
        7: [32,33],
        8: [34,35],
        9: [36,37],
        10:[38,39],
    },
    "2021_10_10":{0: [0,1,2,3,4,5,6,7,8,9],
                  1: [10,11,12,13,14,15,16,17,18,19],
                  2:[20,21,22,23,24,25,26,27,28,29],
                  3:[30,31,32,33,34,35,36,37,38,39],
            },
    'ovis_15_5':{0: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],
                 1: [15, 16, 17, 18, 19],
                 2: [20,21,22,23,24],

            },
    'ovis_15_10':{0: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],
                  1: [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
            },
    }


def get_task_list():
    return list(tasks_vis.keys())


def get_task_labels(dataset, name, step):
    if dataset == 'vis':
        task_dict = tasks_vis[name]
    # elif dataset == 'ade':
    #     task_dict = tasks_ade[name]

    else:
        raise NotImplementedError
    assert step in task_dict.keys(), f"You should provide a valid step! [{step} is out of range]"

    labels = list(task_dict[step])
    if step == 0:
        labels_old=[]
    else:
        labels_old = [label for s in range(step) for label in task_dict[s]]

    return labels, labels_old


def get_per_task_classes(dataset, name, step):
    if dataset == 'vis':
        if name=='vspw':
            total_classes = 124
            base_classes = 74
            inc_classes = 10
            num_steps = int((total_classes - base_classes) / inc_classes)
            task_dict = {0: list(range(base_classes))}
            for i in range(num_steps):
                s = base_classes + i * inc_classes
                e = s + inc_classes
                labels = list(range(s, e))
                task_dict.update({i+1: labels})
            tasks_vis.update({'vspw': task_dict})
        else:
            task_dict = tasks_vis[name]
    # elif dataset == 'ade':
    #     task_dict = tasks_ade[name]
    else:
        raise NotImplementedError
    assert step in task_dict.keys(), f"You should provide a valid step! [{step} is out of range]"

    classes = [len(task_dict[s]) for s in range(step+1)]
    return classes

