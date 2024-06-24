import pandas as pd
import sklearn.model_selection
from tqdm import tqdm
from tracking_policy_agendas.classifiers.pa_clf import PAClf
from tracking_policy_agendas.classifiers.xgb_clf import XgbClf
from tracking_policy_agendas.classifiers.naive_bayes_clf import GNBClf
from tracking_policy_agendas.classifiers.lasso_clf import LassoClf
from tracking_policy_agendas.preprocess.preprocessing import remove_emoji, remove_redundant_characters

tqdm.pandas()


def inference_pipeline(model_path: str, input_text: str):
    xgb = XgbClf(text_array=None, labels=None, load_path=model_path)
    return xgb.predict(input_text)


def main(embedding_frame:pd.DataFrame, dataframe: pd.DataFrame, save_path: str):
    dataframe.replace('', float('NaN')).dropna(inplace=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataframe.text, dataframe.label, stratify=dataframe.label)
    xgb = XgbClf(embedding_doc=embedding_frame.text)
    xgb.fit(X_train, y_train)
    xgb.predict(X_test, y_test)
    xgb.save_model('xgb_' + save_path)
    pa = PAClf(embedding_doc=embedding_frame.text)
    pa.fit(X_train, y_train)
    pa.predict(X_test, y_test)
    pa.save_model('pa_' + save_path)
    lasso = LassoClf(embedding_doc=embedding_frame.text)
    lasso.fit(X_train, y_train)
    lasso.predict(X_test, y_test)
    lasso.save_model('lasso_' + save_path)
    gnb = GNBClf(embedding_doc=embedding_frame.text)
    gnb.fit(X_train, y_train)
    gnb.predict(X_test, y_test)
    gnb.save_model('gnb_' + save_path)


if __name__ == '__main__':
    df = pd.read_excel('vacine_sampling.xlsx')
    # df = pd.read_excel('tweet-zare-relabeled.xlsx')
    # df['label'] = df['polarity-z'].apply(lambda item: int(item == 'positive'))
    # emb_df = pd.read_excel('tweet-zare-relabeled.xlsx')
    emb_df = pd.read_excel('clf_status_id_dummy.xlsx').sample(30000)
    # df['prep_text'] = df.text.progress_apply(lambda item: remove_redundant_characters(remove_emoji(item)))
    # df = df.replace('', float('NaN')).dropna()
    # emb_df['prep_text'] = emb_df.text.progress_apply(lambda item: remove_redundant_characters(remove_emoji(item)))
    # emb_df = emb_df.replace('', float('NaN')).dropna()
    main(emb_df, df, 'vaccine')
