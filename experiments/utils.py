def get_data_id(data_type='num', task_type='regression'):
    num_regression = [197,
                      201,
                      216,
                      300,
                      287,
                      296,
                      537,
                      574,
                      42225,
                      42688,
                      42712,
                      42729,
                      42731,
                      23515,
                      42720,
                      43093,
                      43174,
                      ]

    num_classification = [151,
                          293,
                          354,
                          722,
                          821,
                          993,
                          1120,
                          1461,
                          1489,
                          41150,
                          42769,
                          1044,
                          ]

    cat_classification = [151,
                          1044,
                          1114,
                          1596,
                          41160,
                          42803,
                          ]

    cat_regression = [416,
                      504,
                      688,
                      41540,
                      42225,
                      42571,
                      42570,
                      42688,
                      42712,
                      42729,
                      42731,
                      6331,
                      42207,
                      43144,
                      ]

    if data_type == 'num':
        if task_type == 'regression':
            return num_regression
        elif task_type == 'classification':
            return num_classification
    elif data_type == 'cat':
        if task_type == 'regression':
            return cat_regression
        elif task_type == 'classification':
            return cat_classification
