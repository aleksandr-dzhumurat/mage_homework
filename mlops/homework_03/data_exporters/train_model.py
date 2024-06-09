if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    categorical = ['PULocationID', 'DOLocationID']

    for col in categorical:
        data[col] = data[col].astype('str')
    train_dicts = data[categorical].to_dict(orient='records')

    dv = DictVectorizer()
    dv.fit(train_dicts)
    print('Transformation started')
    X_train = dv.transform(train_dicts)

    print('num cols: %d, num %d' % (X_train.shape[1], X_train.shape[0]))

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    print('Intercept: %.6f' % lr.intercept_)

    return lr, dv

