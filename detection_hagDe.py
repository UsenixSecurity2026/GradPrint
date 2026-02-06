from __future__ import division, absolute_import, print_function
import argparse

from apks import ApkData
import logging
from androguard.core.androconf import show_logging

from common.util import *
from common.draw_sample import *
from core.attack.attack_by_sharpness import attack_label, attack_label_fixed_steps_time

import torch
from torch.utils.data import DataLoader, TensorDataset
from pprint import pformat



def main():
    args = parse_args()
    show_logging(logging.INFO)

    # 设置随机种子
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if args.device_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = ApkData(args.detection, args.granularity, args.classifier, args.attacker, dim_to_be_changed=False,
                      time_test_size=args.time_test_size, args=args)
    before_train__time = time.time()
    if args.Implement_way == 'pytorch':
        from model.nn_pytorch import NnTorch as myModel
        model_save_dir = os.path.join(config['model_save_dir'], args.attacker,
                                      args.detection + (
                                          '_' + args.granularity if args.granularity != '' else '') + '_' + args.classifier + '/')
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir, exist_ok=True)
        model_class = myModel(dataset, model_save_dir,
                              filename='nn_{}_{}_pytorch.h5'.format(args.detection, args.classifier),
                              mode=args.mode, epochs=50, device = device)
        model = model_class.model
        model.to(device)
        model.eval()
    else:
        from model.nn_pytorch import NnTorch as myModel

    after_train__time = time.time()
    model_train_time = after_train__time - before_train__time
    logging.debug("训练模型耗时: {:.2f}秒".format(model_train_time))

    x_train, y_train, x_test, y_test, x_adv, y_adv = \
        dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test, dataset.x_adv, dataset.y_adv

    if args.detection == "drebin" or args.detection == "apigraph":
        x_test = x_test.toarray()
        x_adv = x_adv.toarray()

    logging.debug(green('Test the target model...'))
    x_test_tensor = torch.tensor(x_test).float().to(device)
    x_adv_tensor = torch.tensor(x_adv).float().to(device)
    y_pred = model(x_test_tensor).detach().cpu().numpy()
    accuracy_all = calculate_accuracy(y_pred, y_test)
    # print('Test accuracy on sampled raw legitimate examples %.4f' % (accuracy_all))

    if args.Random_sample:
        random.seed(42)
        sample_num = len(x_adv)
        sample_list = [i for i in range(len(x_test))]
        sample_list = random.sample(sample_list, sample_num)
        x_test = [x_test[i] for i in sample_list]
        y_test = [y_test[i] for i in sample_list]
        x_test_tensor = torch.tensor(x_test).float().to(device)
        y_test_tensor = torch.tensor(y_test).float().to(device)
        test_loader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor),
                                 shuffle=False, batch_size=10)
    else:
        test_loader = dataset.test_loader



    adv_loader = DataLoader(TensorDataset(x_adv_tensor, torch.tensor(y_adv).float()),
                            shuffle=False, batch_size=10)

    detection_classifier_attacker_dir = os.path.join(config['hagDe_result_dir'], "_".join(
        [args.detection, args.granularity, args.classifier, args.attacker]))
    if not os.path.exists(detection_classifier_attacker_dir):
        os.makedirs(detection_classifier_attacker_dir, exist_ok=True)



    if args.performance_fixed_param:
        before_perturb_time = time.time()

        # ======================  Generate perturbation data ========================
        # Settings
        init_mag = 0.005
        lr = 0.032
        perturbation_steps = 16
        detect_type = 'loss'


        # perturb on test sample
        _, test_criteria_all = \
            attack_label_fixed_steps_time(model, test_loader, init_mag, lr, perturbation_steps, detect_type, device)

        # perturb on adv sample
        _, adv_criteria_all = \
            attack_label_fixed_steps_time(model, adv_loader, init_mag, lr, perturbation_steps, detect_type, device)


        X_normal = test_criteria_all.detach().cpu().numpy()
        X_adv = adv_criteria_all.detach().cpu().numpy()

        after_perturb_time = time.time()
        perturb_time = after_perturb_time - before_perturb_time
        logging.info(("扰动耗时: {:.2f}秒".format(perturb_time)))

        y_normal = np.zeros(len(X_normal))
        y_adv = np.ones(len(X_adv))

        X = np.concatenate((X_normal, X_adv))
        Y = np.concatenate((y_normal, y_adv))

        results = train_and_evaluate(X, Y, detection_classifier_attacker_dir, lr, isSave=True, mustRf=True)
        logging.info(blue('Performance of detect:\n' + pformat(results)))

        after_detection_time = time.time()
        detection_time = after_detection_time - before_perturb_time
        # logging.info(("训练分类器耗时: {:.2f}秒".format(detection_time)))

        if args.integrated_feature:
            detection_granularity_dir = args.detection + (
                '_' + args.granularity if args.granularity != '' else '') + '_' + args.classifier
            feature_dir = os.path.join(config['data_source'], args.attacker, 'feature', detection_granularity_dir)
            # X_integrated means feature from lid
            X_integrated = np.load(os.path.join(feature_dir, 'integrated_feature.npy'))
            X_combined = np.concatenate((X, X_integrated), axis=1)
            results = train_and_evaluate(X_combined, Y, detection_classifier_attacker_dir, lr, isSave=False, mustRf=True)
            logging.info(blue('Performance of integrated classifier:\n' + pformat(results)))


    if args.time_consume:
        before_perturb_time = time.time()

        # ======================  Generate perturbation data ========================
        # Settings
        init_mag = 0.005
        lr = 0.032
        perturbation_steps = 16
        detect_type = 'loss'


        # perturb on test sample
        _, test_criteria_all = \
            attack_label_fixed_steps_time(model, dataset.time_test_loader, init_mag, lr, perturbation_steps, detect_type, device)

        # perturb on adv sample
        _, adv_criteria_all = \
            attack_label_fixed_steps_time(model, dataset.time_test_loader, init_mag, lr, perturbation_steps, detect_type, device)

        X_normal = test_criteria_all.detach().cpu().numpy()
        X_adv = adv_criteria_all.detach().cpu().numpy()

        y_normal = np.zeros(len(X_normal))
        y_adv = np.ones(len(X_adv))

        X = np.concatenate((X_normal, X_adv))
        Y = np.concatenate((y_normal, y_adv))

        import joblib
        loaded_rf = joblib.load(os.path.join(detection_classifier_attacker_dir, f'Random Forest_classifier_{X.shape[1]}_{lr}.pkl'))
        defense_y_adv_pred = loaded_rf.predict(X)
        after_detection_time = time.time()
        detection_time = after_detection_time - before_perturb_time
        logging.info(("time cost: {:.2f}秒".format(detection_time)))


    if args.adaptive_attack:
        before_perturb_time = time.time()

        # ======================  Generate perturbation data ========================
        # Settings
        init_mag = 0.005
        lr = 0.032
        perturbation_steps = 16
        detect_type = 'loss'

        # perturb on test sample
        _, test_criteria_all = \
            attack_label_fixed_steps_time(model, test_loader, init_mag, lr, perturbation_steps, detect_type, device)

        # perturb on adv sample
        _, adv_criteria_all = \
            attack_label_fixed_steps_time(model, adv_loader, init_mag, lr, perturbation_steps, detect_type, device)

        X_normal = test_criteria_all.detach().cpu().numpy()
        X_adv = adv_criteria_all.detach().cpu().numpy()

        y_normal = np.zeros(len(X_normal))
        y_adv = np.ones(len(X_adv))

        X = np.concatenate((X_normal, X_adv))
        Y = np.concatenate((y_normal, y_adv))



        # import joblib
        # loaded_rf = joblib.load(os.path.join(detection_classifier_attacker_dir, f'Random Forest_classifier_{X.shape[1]}_{lr}.pkl'))
        # defense_y_adv_pred = loaded_rf.predict(X)
        # defense_y_pred = defense_y_adv_pred.astype(int)
        #
        # report = calculate_base_metrics(Y.astype(int), defense_y_pred, None)
        # report['number_of_apps'] = {'train': len(y_train),
        #                             'test': len(Y),
        #                             }
        # logging.info(blue('Performance of enhanced attack classifier:\n' + pformat(report)))


        # 十折交叉验证
        results = train_and_evaluate(X, Y, detection_classifier_attacker_dir, lr, isSave=False, mustRf=True)
        logging.info(blue('Performance of detect:\n' + pformat(results)))


        # after_detection_time = time.time()
        # detection_time = after_detection_time - before_perturb_time
        # logging.info(("time cost: {:.2f}秒".format(detection_time)))


    # search the param
    if args.do_search_param:
        # ======================  Generate perturbation data ========================
        # Settings
        init_mag = 0.005
        adv_lr = np.around(np.arange(0.04, -0.001, -0.002), decimals=3)
        perturbation_steps = 16
        detect_type = 'loss'
        max_F1 = 0

        f1_harvest_LR = np.zeros([len(adv_lr), perturbation_steps])
        f1_harvest_SVM = np.zeros([len(adv_lr), perturbation_steps])
        f1_harvest_RF = np.zeros([len(adv_lr), perturbation_steps])
        f1_harvest_1NN = np.zeros([len(adv_lr), perturbation_steps])
        f1_harvest_3NN = np.zeros([len(adv_lr), perturbation_steps])

        for i in range(len(adv_lr)):
            lr = adv_lr[i]

            logging.info(magenta('init_mag: %2f, lr: %2f' % (init_mag, lr)))
            logging.info(blue('Begin Detect ------- Building the gradient attack'))

            # perturb on test sample
            perturbation_step_list, test_label_not_flip_rate, test_criteria_all, _ = \
                attack_label(model, test_loader, init_mag, lr, perturbation_steps, detect_type)

            # perturb on adv sample
            _, adv_label_not_flip_rate, adv_criteria_all, _ = \
                attack_label(model, adv_loader, init_mag, lr, perturbation_steps, detect_type)

            X_normal = test_criteria_all.permute(1, 0).detach().numpy()
            X_adv = adv_criteria_all.permute(1, 0).detach().numpy()
            y_normal = np.zeros(len(X_normal))
            y_adv = np.ones(len(X_adv))

            # import joblib
            # loaded_rf = joblib.load("/tmp/pycharm_project_843/results/core/mamadroid__rf_AdvDroidZero/Random Forest_classifier_16_0.032.pkl")
            # defense_y_adv_pred = loaded_rf.predict(test_criteria_all.detach().numpy())

            for step_dim in range(perturbation_steps):
                X_normal_dim = X_normal[:, 0: (step_dim + 1)]
                X_adv_dim = X_adv[:, 0: (step_dim + 1)]

                X = np.concatenate((X_normal_dim, X_adv_dim))
                Y = np.concatenate((y_normal, y_adv))

                results = train_and_evaluate(X, Y, detection_classifier_attacker_dir, lr, isSave=False, mustRf=False)
                logging.info(blue('Performance of detect:\n' + pformat(results)))

                # f1_harvest_LR[i][step_dim] = results['Logistic Regression']['f1']
                f1_harvest_SVM[i][step_dim] = results['SVM']['f1']
                f1_harvest_RF[i][step_dim] = results['Random Forest']['f1']
                f1_harvest_1NN[i][step_dim] = results['1NN']['f1']
                f1_harvest_3NN[i][step_dim] = results['3NN']['f1']


        # 保存所有分类器的结果到一个文件中
        file_name = f"{args.attacker}_{args.detection}_{args.granularity}_{args.classifier}_all_classifiers.npy"
        save_fir = os.path.join(config['hagDe_result_dir'], 'all_classifier2', file_name)

        # 使用字典存储所有分类器的结果
        all_classifiers_results = {
            'SVM': f1_harvest_SVM,
            'RF': f1_harvest_RF,
            '1NN': f1_harvest_1NN,
            '3NN': f1_harvest_3NN
        }

        # 保存到文件
        np.save(save_fir, all_classifiers_results)

        # 加载保存的文件
        loaded_all_classifiers_results = np.load(save_fir, allow_pickle=True).item()

        # 验证加载的数组是否与原始数组匹配
        assert np.array_equal(f1_harvest_SVM, loaded_all_classifiers_results[
            'SVM']), "The loaded SVM array does not match the original array."
        assert np.array_equal(f1_harvest_RF, loaded_all_classifiers_results[
            'RF']), "The loaded RF array does not match the original array."
        assert np.array_equal(f1_harvest_1NN, loaded_all_classifiers_results[
            '1NN']), "The loaded 1NN array does not match the original array."
        assert np.array_equal(f1_harvest_3NN, loaded_all_classifiers_results[
            '3NN']), "The loaded 3NN array does not match the original array."

        print("All arrays saved and loaded successfully.")

        exit()


    # false_positive/false_negative
    if args.enhance_performance:
        # ======================  Generate perturbation data ========================
        # Settings

        # load trained classifier(attacker)
        if os.path.exists(dataset.base_clf_path + ".clf"):
            logging.debug(blue('Loading model from {}...'.format(dataset.base_clf_path + ".clf")))
            with open(dataset.base_clf_path + ".clf", "rb") as f:
                base_clf = pickle.load(f)

        args.detection = args.detection + '_mixed'
        saved_attack_sample_dir = os.path.join(config['data_source'], args.attacker, 'attack_sample', args.detection)

        x_adv_all = np.load(
            os.path.join(saved_attack_sample_dir, args.detection + "_" + args.classifier + "_" + "attack.npy"))
        y_adv_all = np.load(
            os.path.join(saved_attack_sample_dir, args.detection + "_" + args.classifier + "_" + "attack_label.npy"))

        random.seed(42)
        x_test = dataset.x_test
        y_test = dataset.y_test
        sample_num = len(dataset.x_adv)
        sample_list = [i for i in range(len(x_test))]
        sample_used_list = random.sample(sample_list, sample_num)
        x_filter_test = [x_test[i] for i in range(len(x_test)) if i not in sample_used_list]
        y_filter_test = [y_test[i] for i in range(len(y_test)) if i not in sample_used_list]
        x_filter_test = np.asarray(x_filter_test)
        y_filter_test = np.asarray(y_filter_test)
        y_filter_test = np.argmax(y_filter_test, axis=1)

        # 分隔数据
        # 创建一个布尔掩码，用于选择y_filter_test中值为0的元素
        mask_0 = y_filter_test == 0
        mask_1 = y_filter_test == 1

        # 使用布尔掩码来过滤x_filter_test和y_filter_test
        x_filter_test_0 = x_filter_test[mask_0]
        y_filter_test_0 = y_filter_test[mask_0]

        x_filter_test_1 = x_filter_test[mask_1]
        y_filter_test_1 = y_filter_test[mask_1]

        x_filter_adv = dataset.x_adv
        y_filter_adv = np.ones(len(x_filter_adv))

        origin_detect_metric = []
        enhanced_detect_metric = []

        # 定义ration的列表
        fixed_num = 200
        # rations = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        rations = [0.5]
        for ration in rations:
            sample_num = int(fixed_num * ration)
            # 从x_filter_test_1中按照1 - ration随机选择
            selected_x_test_1 = random.sample(list(x_filter_test_1), fixed_num - sample_num)
            selected_y_test_1 = np.ones(len(selected_x_test_1))

            # 从x_filter_adv中按照1 - ration随机选择
            selected_x_adv = random.sample(list(x_adv_all), sample_num)
            selected_y_adv = np.ones(len(selected_x_adv))

            # x_filter_test_0全选
            selected_x_test_0 = random.sample(list(x_filter_test_0), fixed_num)
            selected_y_test_0 = np.zeros(len(selected_x_test_0))

            # 合并所有选中的数据
            x_mixed = selected_x_test_0 + selected_x_test_1 + selected_x_adv
            y_mixed = np.concatenate((selected_y_test_0, selected_y_test_1, selected_y_adv))

            attack_y_pred = base_clf.predict(x_mixed)
            attack_y_score = base_clf.predict_proba(x_mixed)

            report = calculate_base_metrics(y_mixed.astype(int), attack_y_pred, attack_y_score)
            report['number_of_apps'] = {'train': len(y_train),
                                        'test': len(y_mixed)}

            logging.info(blue('Performance of attack classifier:\n' + pformat(report)))
            origin_detect_metric.append(report['model_performance']['f1'])

            init_mag = 0.005
            adv_lr = 0.032
            perturbation_steps = 16

            model.eval()
            mixed_loader = DataLoader(
                TensorDataset(torch.tensor(x_mixed).float(), torch.tensor(y_mixed).float()), shuffle=False,
                batch_size=1)

            # perturb on test sample
            perturbation_step_list, test_label_not_flip_rate, test_criteria_all, y_sum = \
                attack_label(model, mixed_loader, init_mag, adv_lr, perturbation_steps, args.detect_type)

            test_criteria_all = test_criteria_all.permute(1, 0).detach().numpy()

            import joblib
            loaded_rf = joblib.load(
                "/tmp/pycharm_project_843/results/core/mamadroid__rf_AdvDroidZero/Random Forest_classifier_16_0.032.pkl")
            defense_y_adv_pred = loaded_rf.predict(test_criteria_all)

            # sum(np.where(np.asarray(y_sum) == defense_y_adv_pred))

            # false positive（y_sum为0，defense_y_adv_pred为1）
            false_positive_indices = [i for i in range(len(y_sum)) if
                                      y_sum[i] == 0 and attack_y_pred[i] == 0 and defense_y_adv_pred[i] == 1]
            false_positives = [x_mixed[i] for i in false_positive_indices]

            # false_negative（y_sum为1，defense_y_adv_pred为0）
            false_negative_indices = [i for i in range(len(y_sum)) if
                                      y_sum[i] == 1 and attack_y_pred[i] == 0 and defense_y_adv_pred[i] == 0]
            false_negatives = [x_mixed[i] for i in false_negative_indices]

            # Convert to NumPy arrays
            false_positives_array = np.array(false_positives)
            false_negatives_array = np.array(false_negatives)

            # Save as .npy files
            np.save(os.path.join(config['data_source'], 'false_positives.npy'), false_positives_array)
            np.save(os.path.join(config['data_source'], 'false_negatives.npy'), false_negatives_array)

            begin_tps = np.where((attack_y_pred.astype(int) | np.array(defense_y_adv_pred).astype(int)) == 0)[0]
            defense_y_pred = np.ones(len(defense_y_adv_pred)).astype(int)
            for index in begin_tps:
                defense_y_pred[index] = 0

            # 貌似缺少后续一步
            report = calculate_base_metrics(y_mixed.astype(int), defense_y_pred, None)
            report['number_of_apps'] = {'train': len(y_train),
                                        'test': len(y_mixed),
                                        }

            logging.info(blue('Performance of enhanced attack classifier:\n' + pformat(report)))
            enhanced_detect_metric.append(report['model_performance']['f1'])

        draw_enchanced_performance(origin_detect_metric, enhanced_detect_metric, detection_classifier_attacker_dir)

        exit()


def parse_args():
    p = argparse.ArgumentParser()

    # Experiment variables
    p.add_argument('-R', '--run-tag', help='An identifier for this experimental setup/run.')
    p.add_argument('--mode', type=str, default="load", help='Train or load the model.')

    # Choose the target android dataset
    p.add_argument('--dataset', type=str, default="apg", help='The target malware dataset.')

    # Choose the target feature extraction method
    p.add_argument('--detection', type=str, default="mamadroid", help='The target malware feature extraction method.')

    # Choose the granularity
    p.add_argument('--granularity', type=str, default="", help='The target malware feature extraction method.')

    # Choose the target classifier
    p.add_argument('--classifier', type=str, default="rf", help='The target malware classifier.')

    # Choose the attack method
    p.add_argument('--attacker', type=str, default="AdvDroidZero", help='The attack method.')

    # Choose the detect type
    p.add_argument('--detect_type', type=str, default="loss", help='Train or load the model.')

    # Attackers
    p.add_argument('--Implement_way', type=str, default="pytorch", help='Model implement way')
    p.add_argument('--Random_sample', action='store_true', help='randomly sampled test data or not.')
    p.add_argument('--time_consume', action='store_true', help='time consume.')
    p.add_argument('--enhance_performance', action='store_true', help='enhance_performance with/without hagDe false positive & false negative')
    p.add_argument('--do_search_param', action='store_true', help='search for best param.')
    p.add_argument('-T', '--time_test_size', type=int, default=50, help='Different data scale e.g.,5 50 500 * 2')
    p.add_argument('--performance_fixed_param', action='store_true', help='performance.')
    p.add_argument('--integrated_feature', action='store_true', help='whether integrate other features. e.g.,LID')
    p.add_argument('--device_cpu', action='store_true', help='whether run on cpu.')
    p.add_argument('--adaptive_attack', type=str, default=None, help='Attack query and stop success times.')


    # Misc
    p.add_argument('-D', '--debug', action='store_true', help='Display log output in console if True.')

    args = p.parse_args()
    return args


if __name__ == '__main__':
    main()