import featuretools as ft
#
#
# DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
# HOUSING_PATH = os.path.join("datasets", "housing")
# HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
#
#
#
# def fetch_housing_data(housing_url=HOUSING_URL,
#                       housing_path=HOUSING_PATH):
#     if not os.path.isdir(housing_path):
#         os.makedirs(housing_path)
#     tgz_path = os.path.join(housing_path, "housing.tgz")
#     urllib.request.urlretrieve(housing_url, tgz_path)
#     housing_tgz = tarfile.open(tgz_path)
#     housing_tgz.extractall(path=housing_path)
#     housing_tgz.close()
#
# def load_housing_data(housing_path=HOUSING_PATH):
#     csv_path = os.path.join(housing_path, "housing.csv")
#     return pd.read_csv(csv_path)
# fetch_housing_data()
# housing = load_housing_data()
if __name__ == '__main__':
    data = ft.demo.load_mock_customer()

    # x = [4,5,8,12,15]
    # y = [1,2,3,3,2]
    # merge_data = list(zip(x,y))
    # merge_data = sorted(merge_data,key=lambda x: x[0])
    # merge_data = np.array(merge_data).astype(np.int32)
    # temp = entropy_binning(x,y,2)
    # # plt.hist(y,temp)
    # plt.show()
    # print(temp)
