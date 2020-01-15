from SentiCR import SentiCR
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support
import numpy as np
from sklearn.model_selection import train_test_split

#All examples are acutal code review comments from Go lang

sentences=["I'm not sure I entirely understand what you are saying. "+\
           "However, looking at file_linux_test.go I'm pretty sure an interface type would be easier for people to use.",
           "I think it always returns it as 0.",
           "If the steal does not commit, there's no need to clean up _p_'s runq. If it doesn't commit,"+\
             " runqsteal just won't update runqtail, so it won't matter what's in _p_.runq.",
           "Please change the subject: s:internal/syscall/windows:internal/syscall/windows/registry:",
           "I don't think the name Sockaddr is a good choice here, since it means something very different in "+\
           "the C world.  What do you think of SocketConnAddr instead?",
           "could we use sed here? "+\
            " https://go-review.googlesource.com/#/c/10112/1/src/syscall/mkall.sh "+\
            " it will make the location of the build tag consistent across files (always before the package statement).",
           "Is the implementation hiding here important? This would be simpler still as: "+\
          " typedef struct GoSeq {   uint8_t *buf;   size_t off;   size_t len;   size_t cap; } GoSeq;",
           "Make sure you test both ways, or a bug that made it always return false would cause the test to pass. "+\
        " assertTrue(Testpkg.Negate(false)); "+\
        " assertFalse(Testpkg.Negate(true)); +"\
        " If you want to use the assertEquals form, be sure the message makes clear what actually happened and " +\
        "what was expected (e.g. Negate(true) != false). "]

def generate_train_test_sets(dataset_file_path):

    # get training and testing set
    df = pd.read_csv(dataset_file_path)

    X = df['Text']
    y = df['rating']
    seed = np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)

    train_df = pd.DataFrame()
    train_df['rating'] = y_train
    train_df['Text'] = X_train
    # train_df.to_csv("training_set_3000.csv",index=None)

    test_df = pd.DataFrame()
    test_df['rating'] = y_test
    test_df['Text'] = X_test
    # test_df.to_csv("testing_set_3000.csv",index=None)

    return train_df, test_df


def validation(train_df, test_df):
    classifier_model = SentiCR(algo="GBT", training_data=train_df)

    test_comments = test_df['Text'].values
    test_ratings = test_df['rating'].values

    pred = classifier_model.get_sentiment_polarity_collection(test_comments)

    return test_ratings, pred


train_df, test_df = generate_train_test_sets("dataset_3000.csv")
test_ratings, pred = validation(train_df, test_df)
print(classification_report(test_ratings, pred))

# df = pd.read_csv("dataset_3000.csv")
# sentences = df['Text'].values
# pred = []
# test_ratings = df['rating'].values
# print(test_ratings)
# sentiment_analyzer=SentiCR(algo="GBT", training_data=train_df)
# for sent in sentences:
#     score=sentiment_analyzer.get_sentiment_polarity(sent)
#     # print(sent+"\n Score: "+str(score))
#     pred.append(score)
# print(classification_report(test_ratings, pred))