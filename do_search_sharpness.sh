#!/bin/bash
#python detection_GradPrint.py -R detection_GradPrint  --mode load --detection mamadroid --classifier RF --attacker HIV_CW --granularity package --Random_sample --do_search_param >> core.log 2>&1 &&
#python detection_GradPrint.py -R detection_GradPrint  --mode load --detection mamadroid --classifier RF_APIGraph --attacker HIV_CW --granularity package --Random_sample --do_search_param >> core.log 2>&1 &&
#nohup python detection_GradPrint.py -R detection_GradPrint  --mode load --detection mamadroid --classifier RF --attacker HIV_JSMA --granularity package --Random_sample --do_search_param >> core.log 2>&1 &&
nohup python detection_GradPrint.py -R detection_GradPrint  --mode load --detection mamadroid --classifier RF_APIGraph --attacker HIV_JSMA --granularity package --Random_sample --do_search_param >> sharpness.log 2>&1
#python detection_GradPrint.py -R detection_GradPrint --mode load --detection mamadroid --classifier rf --attacker AdvDroidZero --do_search_param --Save_perturb_figure --Random_sample >> core.log 2>&1 &&
#python detection_GradPrint.py -R detection_GradPrint --mode load --detection mamadroid --granularity family --classifier RF --attacker BagAmmo --do_search_param --Save_perturb_figure --Random_sample >> core.log 2>&1 &&
#python detection_GradPrint.py -R detection_GradPrint --mode load --detection mamadroid --granularity family --classifier RF_APIGraph --attacker BagAmmo --do_search_param --Save_perturb_figure --Random_sample >> core.log 2>&1 &&
#python detection_GradPrint.py -R detection_GradPrint --mode load --detection mamadroid --classifier RF --granularity family --attacker HIV_CW --do_search_param --Save_perturb_figure --Random_sample >> core.log 2>&1 &&
#python detection_GradPrint.py -R detection_GradPrint --mode load --detection mamadroid --classifier RF_APIGraph --attacker HIV_CW --granularity family --Random_sample --do_search_param  >> core.log 2>&1 &&
#python detection_GradPrint.py -R detection_GradPrint --mode load --detection mamadroid --classifier RF --granularity family --attacker HIV_JSMA --do_search_param --Save_perturb_figure --Random_sample --do_search_param >> core.log 2>&1 &&
#python detection_GradPrint.py -R detection_GradPrint --mode load --detection mamadroid --classifier RF_APIGraph --attacker HIV_JSMA --granularity family --Random_sample --do_search_param >> core.log 2>&1 &&
#python detection_GradPrint.py -R detection_GradPrint --mode load --detection drebin --classifier svm --attacker AdvDroidZero --do_search_param --Save_perturb_figure --Random_sample >> core.log 2>&1 &&
#python detection_GradPrint.py -R detection_GradPrint --mode load --detection apigraph --classifier svm --attacker AdvDroidZero --do_search_param --Save_perturb_figure --Random_sample >> core.log 2>&1 &&
#python detection_GradPrint.py -R detection_GradPrint --mode load --detection mamadroid --granularity package --classifier RF --attacker BagAmmo --do_search_param --Save_perturb_figure --Random_sample >> core.log 2>&1 &&
#python detection_GradPrint.py -R detection_GradPrint --mode load --detection mamadroid --granularity package --classifier RF_APIGraph --attacker BagAmmo --do_search_param --Save_perturb_figure --Random_sample >> core.log 2>&1
#
