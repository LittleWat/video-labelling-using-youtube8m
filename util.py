def mkdir_if_not_exist(dir_name, verbose=True):
    import os
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        if verbose:
            print (dir_name + " was created!")
    else:
        if verbose:
            print (dir_name + ' was existed...')
