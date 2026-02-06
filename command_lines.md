## GradPrint
## AdvDroidZero
- mamadroid_rf
  - python detection_GradPrint.py -R detection_GradPrint --mode load --detection mamadroid --classifier rf --attacker AdvDroidZero --Random_sample --performance_fixed_param --integrated_feature --device_cpu 

- drebin_svm
  - python detection_GradPrint.py -R detection_GradPrint --mode load --detection drebin --classifier svm --attacker AdvDroidZero --Random_sample --performance_fixed_param --integrated_feature --device_cpu 

- apigraph_svm
  - python detection_GradPrint.py -R detection_GradPrint --mode load --detection apigraph --classifier svm --attacker AdvDroidZero --Random_sample --performance_fixed_param --integrated_feature --device_cpu 


  
## BagAmmo
- mamadroid_family_RF
  - python detection_GradPrint.py -R detection_GradPrint --mode load --detection mamadroid --granularity family --classifier RF --attacker BagAmmo --Random_sample --performance_fixed_param --integrated_feature --device_cpu

- mamadroid_family_RF_APIGraph
  - python detection_GradPrint.py -R detection_GradPrint --mode load --detection mamadroid --granularity family --classifier RF_APIGraph --attacker BagAmmo --Random_sample --performance_fixed_param --integrated_feature --device_cpu

- mamadroid_package_RF
  - python detection_GradPrint.py -R detection_GradPrint --mode load --detection mamadroid --granularity package --classifier RF --attacker BagAmmo --Random_sample --performance_fixed_param --integrated_feature --device_cpu 

- mamadroid_package_RF_APIGraph
  - python detection_GradPrint.py -R detection_GradPrint --mode load --detection mamadroid --granularity package --classifier RF_APIGraph --attacker BagAmmo --Random_sample --performance_fixed_param --integrated_feature --device_cpu


## HIV_CW
- mamadroid_family_RF
  - python detection_GradPrint.py -R detection_GradPrint --mode load --detection mamadroid --granularity family --classifier RF --attacker HIV_CW --Random_sample --performance_fixed_param --integrated_feature --device_cpu

- mamadroid_family_RF_APIGraph
  - python detection_GradPrint.py -R detection_GradPrint --mode load --detection mamadroid --granularity family --classifier RF_APIGraph --attacker HIV_CW --Random_sample --performance_fixed_param --integrated_feature --device_cpu

- mamadroid_package_RF
  - python detection_GradPrint.py -R detection_GradPrint --mode load --detection mamadroid --granularity package --classifier RF --attacker HIV_CW --Random_sample --performance_fixed_param --integrated_feature --device_cpu 

- mamadroid_package_RF_APIGraph
  - python detection_GradPrint.py -R detection_GradPrint --mode load --detection mamadroid --granularity package --classifier RF_APIGraph --attacker HIV_CW --Random_sample --performance_fixed_param --integrated_feature --device_cpu


## HIV_JSMA
- mamadroid_family_RF
  - python detection_GradPrint.py -R detection_GradPrint --mode load --detection mamadroid --granularity family --classifier RF --attacker HIV_JSMA --Random_sample --performance_fixed_param --integrated_feature --device_cpu

- mamadroid_family_RF_APIGraph
  - python detection_GradPrint.py -R detection_GradPrint --mode load --detection mamadroid --granularity family --classifier RF_APIGraph --attacker HIV_JSMA --Random_sample --performance_fixed_param --integrated_feature --device_cpu

- mamadroid_package_RF
  - python detection_GradPrint.py -R detection_GradPrint --mode load --detection mamadroid --granularity package --classifier RF --attacker HIV_JSMA --Random_sample --performance_fixed_param --integrated_feature --device_cpu 

- mamadroid_package_RF_APIGraph
  - python detection_GradPrint.py -R detection_GradPrint --mode load --detection mamadroid --granularity package --classifier RF_APIGraph --attacker HIV_JSMA --Random_sample --performance_fixed_param --integrated_feature --device_cpu




## time_consume
  - python detection_GradPrint.py -R detection_GradPrint --mode load --detection mamadroid --classifier rf --attacker AdvDroidZero --Random_sample --time_consume --device_cpu -T 50
  - python detection_GradPrint.py -R detection_GradPrint --mode load --detection drebin --classifier svm --attacker AdvDroidZero --Random_sample --time_consume --device_cpu -T 5
  - python detection_GradPrint.py -R detection_GradPrint --mode load --detection mamadroid --granularity package --classifier RF --attacker HIV_CW --Random_sample --device_cpu --time_consume -T 50

