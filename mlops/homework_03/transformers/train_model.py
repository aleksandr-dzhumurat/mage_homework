if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LinearRegression

    # Specify your transformation logic here
    categorical = ['PULocationID', 'DOLocationID']
    print(type(data)) 

    for col in categorical:
        data[col] = data[col].astype('str')
    train_dicts = data[categorical].to_dict(orient='records')

    dv = DictVectorizer()
    dv.fit(train_dicts)
    print('Transformation started')
    X_train = dv.transform(train_dicts)

    print('num cols: %d, num %d' % (X_train.shape[1], X_train.shape[0]))

    y_train = (
        (data['tpep_dropoff_datetime'] - data['tpep_pickup_datetime'])
        .dt
        .total_seconds()
        .div(60)
    )

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    print('Intercept: %.6f' % lr.intercept_)

    return lr, dv


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'