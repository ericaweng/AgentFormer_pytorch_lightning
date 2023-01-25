def get_stanford_drone_split():
    train = [
        'bookstore_0', 'bookstore_1', 'bookstore_2', 'bookstore_3', 'coupa_3',
        'deathCircle_0', 'deathCircle_1', 'deathCircle_2', 'deathCircle_3',
        'deathCircle_4', 'gates_0', 'gates_1', 'gates_3', 'gates_4', 'gates_5',
        'gates_6', 'gates_7', 'gates_8', 'hyang_4', 'hyang_5', 'hyang_6',
        'hyang_9', 'nexus_0', 'nexus_1', 'nexus_3', 'nexus_4', 'nexus_7',
        'nexus_8', 'nexus_9'
    ]
    val = []
    # val = [
    #     'coupa_2.txt', 'hyang_10.txt', 'hyang_11.txt', 'hyang_12.txt',
    #     'hyang_13.txt', 'hyang_14.txt', 'hyang_2.txt', 'hyang_7.txt',
    #     'nexus_2.txt', 'nexus_10.txt', 'nexus_11.txt'
    # ]
    test = [
        'coupa_0', 'hyang_3', 'quad_3', 'little_2', 'nexus_5', 'quad_2',
        'gates_2', 'coupa_1', 'quad_1', 'hyang_1', 'hyang_8', 'little_1',
        'nexus_6', 'hyang_0', 'quad_0', 'little_0', 'little_3'
    ]

    return train, val, test
