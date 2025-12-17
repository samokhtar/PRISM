from data_provider.data_loader import building_design

def get_data(args):
    data_dict = {
        'building_design': building_design,
    }
    dataset = data_dict[args.loader](args)
    train_loader, val_loader, test_loader, real_loader, shapelist = dataset.get_loader()
    return dataset, train_loader, val_loader, test_loader, real_loader, shapelist
