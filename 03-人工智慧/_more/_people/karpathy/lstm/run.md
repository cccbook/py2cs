(minGPT) PS D:\ccc\ai\_diy\karpathy\more> python lstm.py
n= 1 b= 3 d= 4
n= 1 b= 3 d= 4
n= 1 b= 3 d= 4
n= 1 b= 3 d= 4
n= 1 b= 3 d= 4
n= 5 b= 3 d= 4
Making sure batched version agrees with sequential version: (should all be True)
True
True
True
True
Traceback (most recent call last):
  File "D:\ccc\ai\_diy\karpathy\more\lstm.py", line 268, in <module>
    raw_input('check OK, press key to continue to gradient check')
NameError: name 'raw_input' is not defined
(minGPT) PS D:\ccc\ai\_diy\karpathy\more> python lstm.py
n= 1 b= 3 d= 4
n= 1 b= 3 d= 4
n= 1 b= 3 d= 4
n= 1 b= 3 d= 4
n= 1 b= 3 d= 4
n= 5 b= 3 d= 4
Making sure batched version agrees with sequential version: (should all be True)
True
True
True
True
check OK, press key to continue to gradient check
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 0, 0) (val = -0.130340), analytic = +0.034866, numerical = +0.034866, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 0, 1) (val = -0.317361), analytic = +0.072927, numerical = +0.072927, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 0, 2) (val = -0.634345), analytic = +0.024143, numerical = +0.024143, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 0, 3) (val = +0.295091), analytic = +0.089522, numerical = +0.089522, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 0, 4) (val = +0.975936), analytic = +0.000162, numerical = +0.000162, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 0, 5) (val = +1.441657), analytic = +0.061600, numerical = +0.061600, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 0, 6) (val = -0.346071), analytic = +0.013507, numerical = +0.013507, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 0, 7) (val = +2.252653), analytic = +0.012859, numerical = +0.012859, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 0, 8) (val = -0.295130), analytic = -0.001290, numerical = -0.001290, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 0, 9) (val = -0.777905), analytic = -0.001993, numerical = -0.001993, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 1, 0) (val = +0.060376), analytic = +0.358973, numerical = +0.358973, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 1, 1) (val = +1.165887), analytic = +0.160747, numerical = +0.160747, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 1, 2) (val = +1.169160), analytic = -0.341298, numerical = -0.341298, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 1, 3) (val = +0.454615), analytic = -0.131047, numerical = -0.131047, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 1, 4) (val = -0.985833), analytic = -0.291209, numerical = -0.291209, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 1, 5) (val = -0.383921), analytic = +0.086766, numerical = +0.086766, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 1, 6) (val = -0.990061), analytic = -0.128345, numerical = -0.128345, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 1, 7) (val = +1.301305), analytic = -0.007030, numerical = -0.007030, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 1, 8) (val = -1.640197), analytic = -0.370932, numerical = -0.370932, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 1, 9) (val = -0.142443), analytic = +0.005838, numerical = +0.005838, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 2, 0) (val = +1.132496), analytic = -0.076806, numerical = -0.076806, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 2, 1) (val = +0.037788), analytic = -0.162243, numerical = -0.162243, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 2, 2) (val = +1.125697), analytic = +0.223994, numerical = +0.223994, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 2, 3) (val = +0.643805), analytic = +0.014434, numerical = +0.014434, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 2, 4) (val = -1.263902), analytic = -0.076261, numerical = -0.076261, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 2, 5) (val = +0.854381), analytic = +0.128504, numerical = +0.128504, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 2, 6) (val = -0.105851), analytic = +0.185016, numerical = +0.185016, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 2, 7) (val = -0.252946), analytic = -0.081312, numerical = -0.081312, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 2, 8) (val = -1.316421), analytic = +0.247962, numerical = +0.247962, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (0, 2, 9) (val = +2.009751), analytic = +0.108890, numerical = +0.108890, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 0, 0) (val = +0.648986), analytic = +0.107544, numerical = +0.107544, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 0, 1) (val = +0.236713), analytic = +0.130094, numerical = +0.130094, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 0, 2) (val = -0.131506), analytic = +0.072867, numerical = +0.072867, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 0, 3) (val = +0.029854), analytic = -0.082010, numerical = -0.082010, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 0, 4) (val = -0.709894), analytic = -0.099735, numerical = -0.099735, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 0, 5) (val = +0.333803), analytic = +0.184148, numerical = +0.184148, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 0, 6) (val = -0.112818), analytic = +0.034147, numerical = +0.034147, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 0, 7) (val = -0.278236), analytic = +0.067529, numerical = +0.067529, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 0, 8) (val = -0.166184), analytic = -0.052561, numerical = -0.052561, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 0, 9) (val = -1.427974), analytic = +0.126990, numerical = +0.126990, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 1, 0) (val = +0.515127), analytic = +0.124711, numerical = +0.124711, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 1, 1) (val = +0.552033), analytic = +0.186213, numerical = +0.186213, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 1, 2) (val = +0.257200), analytic = -0.120132, numerical = -0.120132, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 1, 3) (val = +1.267885), analytic = -0.040307, numerical = -0.040307, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 1, 4) (val = -1.838223), analytic = -0.087544, numerical = -0.087544, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 1, 5) (val = -1.748797), analytic = +0.119179, numerical = +0.119179, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 1, 6) (val = +0.085780), analytic = -0.076953, numerical = -0.076953, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 1, 7) (val = -0.816602), analytic = +0.199198, numerical = +0.199198, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 1, 8) (val = +0.138818), analytic = -0.145272, numerical = -0.145272, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 1, 9) (val = -0.469967), analytic = +0.001949, numerical = +0.001949, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 2, 0) (val = -0.504593), analytic = -0.082123, numerical = -0.082123, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 2, 1) (val = +1.095194), analytic = +0.055418, numerical = +0.055418, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 2, 2) (val = -0.507665), analytic = +0.302896, numerical = +0.302896, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 2, 3) (val = +1.462475), analytic = -0.202395, numerical = -0.202395, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 2, 4) (val = -0.472327), analytic = -0.108880, numerical = -0.108880, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 2, 5) (val = -0.911619), analytic = +0.279640, numerical = +0.279640, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 2, 6) (val = -0.437078), analytic = +0.287224, numerical = +0.287224, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 2, 7) (val = +0.836845), analytic = +0.258210, numerical = +0.258210, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 2, 8) (val = -0.133866), analytic = +0.266848, numerical = +0.266848, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (1, 2, 9) (val = -0.509341), analytic = +0.185857, numerical = +0.185857, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 0, 0) (val = -0.728665), analytic = +0.127124, numerical = +0.127124, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 0, 1) (val = +0.990758), analytic = +0.082384, numerical = +0.082384, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 0, 2) (val = -0.239972), analytic = +0.057418, numerical = +0.057418, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 0, 3) (val = +0.786364), analytic = -0.030405, numerical = -0.030405, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 0, 4) (val = -0.610338), analytic = -0.117060, numerical = -0.117060, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 0, 5) (val = +0.809692), analytic = +0.130895, numerical = +0.130895, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 0, 6) (val = -0.478811), analytic = +0.116318, numerical = +0.116318, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 0, 7) (val = -1.018752), analytic = +0.155240, numerical = +0.155240, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 0, 8) (val = -0.345991), analytic = -0.093710, numerical = -0.093710, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 0, 9) (val = +0.403765), analytic = +0.071003, numerical = +0.071003, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 1, 0) (val = -0.728850), analytic = -0.043247, numerical = -0.043247, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 1, 1) (val = +0.028087), analytic = -0.126766, numerical = -0.126766, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 1, 2) (val = +0.675794), analytic = -0.120125, numerical = -0.120125, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 1, 3) (val = +2.902450), analytic = +0.129771, numerical = +0.129771, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 1, 4) (val = -0.639803), analytic = -0.171707, numerical = -0.171707, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 1, 5) (val = -0.301041), analytic = -0.171406, numerical = -0.171406, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 1, 6) (val = +0.254041), analytic = +0.007476, numerical = +0.007476, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 1, 7) (val = -2.285232), analytic = +0.187109, numerical = +0.187109, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 1, 8) (val = +2.115154), analytic = +0.048353, numerical = +0.048353, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 1, 9) (val = -0.664914), analytic = -0.195666, numerical = -0.195666, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 2, 0) (val = +0.271752), analytic = +0.060445, numerical = +0.060445, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 2, 1) (val = +0.455263), analytic = +0.013842, numerical = +0.013842, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 2, 2) (val = +0.017144), analytic = +0.242247, numerical = +0.242247, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 2, 3) (val = +1.100051), analytic = +0.047293, numerical = +0.047293, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 2, 4) (val = -0.307858), analytic = -0.037511, numerical = -0.037511, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 2, 5) (val = +1.724234), analytic = +0.263795, numerical = +0.263795, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 2, 6) (val = -0.048413), analytic = +0.160030, numerical = +0.160030, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 2, 7) (val = -0.961496), analytic = -0.023302, numerical = -0.023302, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 2, 8) (val = +0.649540), analytic = -0.004254, numerical = -0.004254, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (2, 2, 9) (val = +1.170384), analytic = +0.137570, numerical = +0.137570, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 0, 0) (val = +0.243459), analytic = +0.008978, numerical = +0.008978, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 0, 1) (val = +1.312732), analytic = -0.048042, numerical = -0.048042, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 0, 2) (val = -1.001493), analytic = -0.122703, numerical = -0.122703, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 0, 3) (val = +0.520194), analytic = -0.034412, numerical = -0.034412, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 0, 4) (val = -1.070866), analytic = -0.059738, numerical = -0.059738, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 0, 5) (val = -0.000232), analytic = -0.048395, numerical = -0.048395, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 0, 6) (val = -0.015442), analytic = -0.064440, numerical = -0.064440, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 0, 7) (val = +0.968446), analytic = -0.196365, numerical = -0.196365, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 0, 8) (val = -0.586580), analytic = +0.099145, numerical = +0.099145, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 0, 9) (val = -1.500452), analytic = -0.020270, numerical = -0.020270, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 1, 0) (val = -0.289153), analytic = +0.018013, numerical = +0.018013, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 1, 1) (val = +0.394035), analytic = +0.029282, numerical = +0.029282, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 1, 2) (val = -0.304055), analytic = -0.158568, numerical = -0.158568, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 1, 3) (val = +0.337796), analytic = +0.040075, numerical = +0.040075, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 1, 4) (val = +0.019518), analytic = +0.006623, numerical = +0.006623, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 1, 5) (val = -0.735914), analytic = -0.136678, numerical = -0.136678, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 1, 6) (val = +0.331274), analytic = -0.153432, numerical = -0.153432, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 1, 7) (val = -0.971855), analytic = -0.006896, numerical = -0.006896, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 1, 8) (val = +0.804694), analytic = -0.090674, numerical = -0.090674, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 1, 9) (val = -0.007867), analytic = -0.087304, numerical = -0.087304, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 2, 0) (val = -0.102924), analytic = +0.015127, numerical = +0.015127, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 2, 1) (val = -0.984824), analytic = -0.059819, numerical = -0.059819, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 2, 2) (val = -1.103966), analytic = -0.020630, numerical = -0.020630, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 2, 3) (val = +0.775221), analytic = -0.138174, numerical = -0.138174, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 2, 4) (val = +0.972931), analytic = -0.027771, numerical = -0.027771, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 2, 5) (val = -0.149734), analytic = +0.091816, numerical = +0.091816, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 2, 6) (val = -1.681376), analytic = +0.006565, numerical = +0.006565, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 2, 7) (val = -1.568193), analytic = -0.161030, numerical = -0.161030, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 2, 8) (val = +0.361224), analytic = +0.098655, numerical = +0.098655, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (3, 2, 9) (val = -0.387257), analytic = +0.168967, numerical = +0.168967, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 0, 0) (val = -0.250359), analytic = -0.022380, numerical = -0.022380, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 0, 1) (val = -0.978061), analytic = -0.054081, numerical = -0.054081, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 0, 2) (val = -0.696757), analytic = -0.121034, numerical = -0.121034, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 0, 3) (val = +1.078468), analytic = +0.052374, numerical = +0.052374, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 0, 4) (val = +0.556830), analytic = +0.001372, numerical = +0.001372, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 0, 5) (val = +1.189737), analytic = -0.140510, numerical = -0.140510, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 0, 6) (val = -0.131287), analytic = -0.100749, numerical = -0.100749, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 0, 7) (val = +0.732797), analytic = -0.105601, numerical = -0.105601, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 0, 8) (val = -0.597108), analytic = +0.012341, numerical = +0.012341, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 0, 9) (val = -0.328186), analytic = -0.083177, numerical = -0.083177, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 1, 0) (val = +0.135135), analytic = -0.089720, numerical = -0.089720, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 1, 1) (val = +0.334918), analytic = +0.169364, numerical = +0.169364, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 1, 2) (val = -1.193084), analytic = +0.031984, numerical = +0.031984, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 1, 3) (val = -1.236500), analytic = +0.000021, numerical = +0.000021, relative error = +0.000001
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 1, 4) (val = +0.784974), analytic = +0.137561, numerical = +0.137561, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 1, 5) (val = -3.216761), analytic = +0.053667, numerical = +0.053667, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 1, 6) (val = +1.544515), analytic = -0.164057, numerical = -0.164057, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 1, 7) (val = -0.610258), analytic = +0.007237, numerical = +0.007237, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 1, 8) (val = -0.100320), analytic = +0.008332, numerical = +0.008332, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 1, 9) (val = -1.773340), analytic = +0.056434, numerical = +0.056434, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 2, 0) (val = +1.044394), analytic = +0.032653, numerical = +0.032653, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 2, 1) (val = -2.379211), analytic = -0.012393, numerical = -0.012393, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 2, 2) (val = +0.506200), analytic = +0.104320, numerical = +0.104320, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 2, 3) (val = -1.060468), analytic = -0.170186, numerical = -0.170186, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 2, 4) (val = -0.311032), analytic = -0.046311, numerical = -0.046311, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 2, 5) (val = -0.986062), analytic = +0.150110, numerical = +0.150110, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 2, 6) (val = +0.236103), analytic = +0.067995, numerical = +0.067995, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 2, 7) (val = +1.206209), analytic = +0.029640, numerical = +0.029640, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 2, 8) (val = +0.511739), analytic = -0.036653, numerical = -0.036653, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param X index (4, 2, 9) (val = +0.540995), analytic = +0.080885, numerical = +0.080885, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (0, 0) (val = +0.000000), analytic = +0.212813, numerical = +0.212813, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (0, 1) (val = +0.000000), analytic = +0.070312, numerical = +0.070312, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (0, 2) (val = +0.000000), analytic = -0.148411, numerical = -0.148411, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (0, 3) (val = +0.000000), analytic = +0.033538, numerical = +0.033538, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (0, 4) (val = +3.000000), analytic = +0.187453, numerical = +0.187453, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (0, 5) (val = +3.000000), analytic = +0.021232, numerical = +0.021232, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (0, 6) (val = +3.000000), analytic = +0.029797, numerical = +0.029797, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (0, 7) (val = +3.000000), analytic = +0.019895, numerical = +0.019895, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (0, 8) (val = +0.000000), analytic = +0.220602, numerical = +0.220602, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (0, 9) (val = +0.000000), analytic = -0.359235, numerical = -0.359235, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (0, 10) (val = +0.000000), analytic = -0.423942, numerical = -0.423942, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (0, 11) (val = +0.000000), analytic = -0.037424, numerical = -0.037424, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (0, 12) (val = +0.000000), analytic = +0.158467, numerical = +0.158467, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (0, 13) (val = +0.000000), analytic = +2.486658, numerical = +2.486658, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (0, 14) (val = +0.000000), analytic = -1.371353, numerical = -1.371353, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (0, 15) (val = +0.000000), analytic = -0.181999, numerical = -0.181999, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (1, 0) (val = -0.035929), analytic = +0.235823, numerical = +0.235823, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (1, 1) (val = -0.039784), analytic = +0.111666, numerical = +0.111666, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (1, 2) (val = -0.321149), analytic = +0.075649, numerical = +0.075649, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (1, 3) (val = +0.211645), analytic = +0.089029, numerical = +0.089029, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (1, 4) (val = -0.366206), analytic = -0.015857, numerical = -0.015857, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (1, 5) (val = -0.334634), analytic = +0.002916, numerical = +0.002916, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (1, 6) (val = -0.768439), analytic = -0.029179, numerical = -0.029179, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (1, 7) (val = -0.162261), analytic = -0.015242, numerical = -0.015242, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (1, 8) (val = -0.142008), analytic = +0.327785, numerical = +0.327785, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (1, 9) (val = -0.022358), analytic = +0.365091, numerical = +0.365091, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (1, 10) (val = +0.478892), analytic = +0.000939, numerical = +0.000939, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (1, 11) (val = -0.313053), analytic = -0.115803, numerical = -0.115803, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (1, 12) (val = +0.145018), analytic = +0.551922, numerical = +0.551922, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (1, 13) (val = +0.245735), analytic = +0.115701, numerical = +0.115701, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (1, 14) (val = -0.102938), analytic = -0.621305, numerical = -0.621305, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (1, 15) (val = -0.390753), analytic = +0.209435, numerical = +0.209435, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (2, 0) (val = -0.422622), analytic = -0.264213, numerical = -0.264213, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (2, 1) (val = -0.146869), analytic = +0.054331, numerical = +0.054331, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (2, 2) (val = +0.076933), analytic = -0.190529, numerical = -0.190529, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (2, 3) (val = +0.408612), analytic = +0.092707, numerical = +0.092707, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (2, 4) (val = -0.555208), analytic = +0.126665, numerical = +0.126665, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (2, 5) (val = -0.169543), analytic = +0.014067, numerical = +0.014067, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (2, 6) (val = -0.297806), analytic = +0.046738, numerical = +0.046738, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (2, 7) (val = -0.147275), analytic = +0.042720, numerical = +0.042720, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (2, 8) (val = +0.235919), analytic = -0.603151, numerical = -0.603151, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (2, 9) (val = +0.351255), analytic = -0.154522, numerical = -0.154522, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (2, 10) (val = -0.219479), analytic = -0.132597, numerical = -0.132597, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (2, 11) (val = +0.132503), analytic = +0.134979, numerical = +0.134979, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (2, 12) (val = -0.199323), analytic = -0.744511, numerical = -0.744511, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (2, 13) (val = +0.310009), analytic = +1.806391, numerical = +1.806391, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (2, 14) (val = +0.073008), analytic = -0.166408, numerical = -0.166408, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (2, 15) (val = -0.006897), analytic = -0.380540, numerical = -0.380540, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (3, 0) (val = +0.317328), analytic = +0.233463, numerical = +0.233463, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (3, 1) (val = +0.585774), analytic = +0.093738, numerical = +0.093738, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (3, 2) (val = -0.044034), analytic = +0.026947, numerical = +0.026947, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (3, 3) (val = -0.143306), analytic = -0.129267, numerical = -0.129267, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (3, 4) (val = +0.004853), analytic = -0.060493, numerical = -0.060493, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (3, 5) (val = +0.080926), analytic = -0.018921, numerical = -0.018921, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (3, 6) (val = -0.873557), analytic = +0.064442, numerical = +0.064442, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (3, 7) (val = +0.058435), analytic = -0.013339, numerical = -0.013339, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (3, 8) (val = +0.034313), analytic = +0.017920, numerical = +0.017920, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (3, 9) (val = +0.192081), analytic = -0.738348, numerical = -0.738348, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (3, 10) (val = -0.176417), analytic = +0.314205, numerical = +0.314205, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (3, 11) (val = -0.427390), analytic = -0.213475, numerical = -0.213475, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (3, 12) (val = +0.065522), analytic = -0.353267, numerical = -0.353267, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (3, 13) (val = +0.104853), analytic = +0.706829, numerical = +0.706829, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (3, 14) (val = +0.170840), analytic = -0.064472, numerical = -0.064472, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (3, 15) (val = +0.544047), analytic = -0.324208, numerical = -0.324208, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (4, 0) (val = +0.159562), analytic = -0.076099, numerical = -0.076099, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (4, 1) (val = +0.522738), analytic = -0.278205, numerical = -0.278205, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (4, 2) (val = +0.204323), analytic = -0.140051, numerical = -0.140051, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (4, 3) (val = +0.136944), analytic = -0.293708, numerical = -0.293708, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (4, 4) (val = -0.400020), analytic = +0.193322, numerical = +0.193322, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (4, 5) (val = +0.253660), analytic = +0.043637, numerical = +0.043637, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (4, 6) (val = +0.030429), analytic = -0.030005, numerical = -0.030005, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (4, 7) (val = +0.629437), analytic = -0.031813, numerical = -0.031813, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (4, 8) (val = -0.379607), analytic = -0.586578, numerical = -0.586578, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (4, 9) (val = -0.221954), analytic = -1.905406, numerical = -1.905406, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (4, 10) (val = -0.356893), analytic = -0.334390, numerical = -0.334390, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (4, 11) (val = -0.069904), analytic = -0.100884, numerical = -0.100884, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (4, 12) (val = -0.174678), analytic = -0.414663, numerical = -0.414663, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (4, 13) (val = -0.180574), analytic = +1.880631, numerical = +1.880631, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (4, 14) (val = +0.078057), analytic = -0.680811, numerical = -0.680811, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (4, 15) (val = +0.057225), analytic = +0.327609, numerical = +0.327609, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (5, 0) (val = +0.225279), analytic = -0.077043, numerical = -0.077043, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (5, 1) (val = +0.032550), analytic = +0.098629, numerical = +0.098629, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (5, 2) (val = -0.144682), analytic = +0.137803, numerical = +0.137803, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (5, 3) (val = -0.016851), analytic = -0.081371, numerical = -0.081371, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (5, 4) (val = +0.298648), analytic = -0.098006, numerical = -0.098006, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (5, 5) (val = +0.141422), analytic = -0.014818, numerical = -0.014818, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (5, 6) (val = -0.211287), analytic = -0.023392, numerical = -0.023392, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (5, 7) (val = -0.219969), analytic = +0.015703, numerical = +0.015703, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (5, 8) (val = +0.066737), analytic = -0.195720, numerical = -0.195720, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (5, 9) (val = +0.392743), analytic = +0.871388, numerical = +0.871388, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (5, 10) (val = -0.082898), analytic = +0.071008, numerical = +0.071008, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (5, 11) (val = +0.250481), analytic = +0.290623, numerical = +0.290623, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (5, 12) (val = -0.041523), analytic = +0.111178, numerical = +0.111178, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (5, 13) (val = -0.228816), analytic = -1.908855, numerical = -1.908855, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (5, 14) (val = +0.283454), analytic = +0.808787, numerical = +0.808787, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (5, 15) (val = +0.179026), analytic = +0.139426, numerical = +0.139426, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (6, 0) (val = +0.459457), analytic = +0.105807, numerical = +0.105807, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (6, 1) (val = -0.190912), analytic = +0.710194, numerical = +0.710194, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (6, 2) (val = +0.029323), analytic = +0.193899, numerical = +0.193899, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (6, 3) (val = +0.322022), analytic = -0.446433, numerical = -0.446433, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (6, 4) (val = +0.128616), analytic = -0.107680, numerical = -0.107680, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (6, 5) (val = +0.174929), analytic = -0.045611, numerical = -0.045611, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (6, 6) (val = -0.222322), analytic = -0.094310, numerical = -0.094310, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (6, 7) (val = -0.113119), analytic = -0.112134, numerical = -0.112134, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (6, 8) (val = -0.064522), analytic = -1.338081, numerical = -1.338081, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (6, 9) (val = +0.309512), analytic = -0.314902, numerical = -0.314902, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (6, 10) (val = +0.061520), analytic = +0.567900, numerical = +0.567900, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (6, 11) (val = -0.327285), analytic = +0.129935, numerical = +0.129935, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (6, 12) (val = +0.190525), analytic = +0.290771, numerical = +0.290771, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (6, 13) (val = +0.400207), analytic = -0.180341, numerical = -0.180341, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (6, 14) (val = -0.073336), analytic = -0.708855, numerical = -0.708855, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (6, 15) (val = +0.237949), analytic = +0.721141, numerical = +0.721141, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (7, 0) (val = -0.205243), analytic = +0.117180, numerical = +0.117180, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (7, 1) (val = +0.411807), analytic = +0.021831, numerical = +0.021831, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (7, 2) (val = +0.133716), analytic = -0.029931, numerical = -0.029931, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (7, 3) (val = +0.190104), analytic = +0.216154, numerical = +0.216154, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (7, 4) (val = +0.036498), analytic = -0.047485, numerical = -0.047485, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (7, 5) (val = +0.015393), analytic = +0.003420, numerical = +0.003420, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (7, 6) (val = +0.100022), analytic = -0.076434, numerical = -0.076434, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (7, 7) (val = -0.000375), analytic = +0.043188, numerical = +0.043188, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (7, 8) (val = +0.001057), analytic = +0.207384, numerical = +0.207384, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (7, 9) (val = -0.315250), analytic = +0.217437, numerical = +0.217437, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (7, 10) (val = +0.487328), analytic = -0.314414, numerical = -0.314414, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (7, 11) (val = +0.042330), analytic = +0.073082, numerical = +0.073082, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (7, 12) (val = +0.097798), analytic = -0.650731, numerical = -0.650731, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (7, 13) (val = +0.116264), analytic = -1.078219, numerical = -1.078219, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (7, 14) (val = -0.055362), analytic = +0.549794, numerical = +0.549794, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (7, 15) (val = +0.425676), analytic = +0.106896, numerical = +0.106896, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (8, 0) (val = -0.219468), analytic = -0.066506, numerical = -0.066506, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (8, 1) (val = +0.285741), analytic = +0.205655, numerical = +0.205655, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (8, 2) (val = -0.088253), analytic = -0.358414, numerical = -0.358414, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (8, 3) (val = +0.385258), analytic = +0.065285, numerical = +0.065285, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (8, 4) (val = +0.035782), analytic = +0.007672, numerical = +0.007672, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (8, 5) (val = -0.121158), analytic = -0.033341, numerical = -0.033341, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (8, 6) (val = -0.351089), analytic = +0.153525, numerical = +0.153525, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (8, 7) (val = +0.244333), analytic = -0.005510, numerical = -0.005510, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (8, 8) (val = +0.412416), analytic = +0.167207, numerical = +0.167207, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (8, 9) (val = -0.413889), analytic = +0.899743, numerical = +0.899743, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (8, 10) (val = -0.086519), analytic = +0.442811, numerical = +0.442811, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (8, 11) (val = +0.028725), analytic = -0.117014, numerical = -0.117014, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (8, 12) (val = -0.172105), analytic = -0.485396, numerical = -0.485396, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (8, 13) (val = +0.313763), analytic = +0.282905, numerical = +0.282905, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (8, 14) (val = +0.470528), analytic = -0.756315, numerical = -0.756315, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (8, 15) (val = +0.249499), analytic = -0.063513, numerical = -0.063513, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (9, 0) (val = -0.197591), analytic = -0.034538, numerical = -0.034538, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (9, 1) (val = -0.261873), analytic = -0.290673, numerical = -0.290673, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (9, 2) (val = -0.086256), analytic = +0.090552, numerical = +0.090552, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (9, 3) (val = -0.556946), analytic = +0.009421, numerical = +0.009421, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (9, 4) (val = -0.019574), analytic = -0.014895, numerical = -0.014895, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (9, 5) (val = +0.306524), analytic = +0.026209, numerical = +0.026209, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (9, 6) (val = +0.031730), analytic = -0.085888, numerical = -0.085888, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (9, 7) (val = +0.703123), analytic = -0.019309, numerical = -0.019309, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (9, 8) (val = -0.069727), analytic = +0.041085, numerical = +0.041085, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (9, 9) (val = +0.024573), analytic = -0.756745, numerical = -0.756745, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (9, 10) (val = -0.445550), analytic = -0.242582, numerical = -0.242582, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (9, 11) (val = +0.289030), analytic = -0.142518, numerical = -0.142518, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (9, 12) (val = +0.048921), analytic = -0.154625, numerical = -0.154625, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (9, 13) (val = -0.127551), analytic = -0.955321, numerical = -0.955321, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (9, 14) (val = -0.201819), analytic = +0.909697, numerical = +0.909697, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (9, 15) (val = +0.474441), analytic = +0.435591, numerical = +0.435591, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (10, 0) (val = +0.053015), analytic = +0.279212, numerical = +0.279212, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (10, 1) (val = -0.209837), analytic = +0.425249, numerical = +0.425249, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (10, 2) (val = +0.145423), analytic = +0.308942, numerical = +0.308942, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (10, 3) (val = -0.197968), analytic = -0.349066, numerical = -0.349066, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (10, 4) (val = +0.191655), analytic = -0.049039, numerical = -0.049039, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (10, 5) (val = -0.157442), analytic = +0.025788, numerical = +0.025788, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (10, 6) (val = -0.275616), analytic = -0.083349, numerical = -0.083349, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (10, 7) (val = +0.165868), analytic = -0.065729, numerical = -0.065729, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (10, 8) (val = +0.042775), analytic = -0.168737, numerical = -0.168737, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (10, 9) (val = +0.344069), analytic = -0.261450, numerical = -0.261450, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (10, 10) (val = +0.022370), analytic = +0.555253, numerical = +0.555253, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (10, 11) (val = +0.071508), analytic = -0.210123, numerical = -0.210123, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (10, 12) (val = +0.277064), analytic = +0.673798, numerical = +0.673798, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (10, 13) (val = +0.234366), analytic = -0.267188, numerical = -0.267188, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (10, 14) (val = -0.059295), analytic = -0.278457, numerical = -0.278457, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (10, 15) (val = +0.119157), analytic = +1.007559, numerical = +1.007559, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (11, 0) (val = +0.301899), analytic = -0.026624, numerical = -0.026624, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (11, 1) (val = -0.290446), analytic = +0.023680, numerical = +0.023680, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (11, 2) (val = -0.297754), analytic = +0.117903, numerical = +0.117903, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (11, 3) (val = -0.116661), analytic = -0.145901, numerical = -0.145901, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (11, 4) (val = +0.030510), analytic = +0.057186, numerical = +0.057186, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (11, 5) (val = +0.368806), analytic = +0.026736, numerical = +0.026736, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (11, 6) (val = -0.034075), analytic = -0.182539, numerical = -0.182539, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (11, 7) (val = +0.097474), analytic = -0.049389, numerical = -0.049389, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (11, 8) (val = +0.162049), analytic = +0.014524, numerical = +0.014524, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (11, 9) (val = +0.156712), analytic = -0.045670, numerical = -0.045670, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (11, 10) (val = -0.527080), analytic = -0.149034, numerical = -0.149034, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (11, 11) (val = +0.171589), analytic = -0.053113, numerical = -0.053113, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (11, 12) (val = -0.131311), analytic = +0.501613, numerical = +0.501613, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (11, 13) (val = +0.082784), analytic = -0.648340, numerical = -0.648340, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (11, 14) (val = +0.012680), analytic = -0.125433, numerical = -0.125433, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (11, 15) (val = -0.196484), analytic = +1.455379, numerical = +1.455379, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (12, 0) (val = -0.196921), analytic = +0.009831, numerical = +0.009831, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (12, 1) (val = -0.151636), analytic = -0.011917, numerical = -0.011917, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (12, 2) (val = -0.396233), analytic = +0.052949, numerical = +0.052949, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (12, 3) (val = +0.003132), analytic = -0.149946, numerical = -0.149946, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (12, 4) (val = -0.173888), analytic = +0.052732, numerical = +0.052732, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (12, 5) (val = -0.321945), analytic = +0.020809, numerical = +0.020809, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (12, 6) (val = +0.094999), analytic = -0.027199, numerical = -0.027199, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (12, 7) (val = -0.097894), analytic = -0.029208, numerical = -0.029208, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (12, 8) (val = +0.077846), analytic = +0.092639, numerical = +0.092639, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (12, 9) (val = +0.425432), analytic = -0.149885, numerical = -0.149885, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (12, 10) (val = -0.077554), analytic = +0.069419, numerical = +0.069419, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (12, 11) (val = +0.431494), analytic = -0.147190, numerical = -0.147190, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (12, 12) (val = +0.112693), analytic = +0.450895, numerical = +0.450895, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (12, 13) (val = -0.444700), analytic = +0.180160, numerical = +0.180160, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (12, 14) (val = -0.051098), analytic = -0.402074, numerical = -0.402074, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (12, 15) (val = +0.016726), analytic = +0.614452, numerical = +0.614452, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (13, 0) (val = +0.130853), analytic = -0.141194, numerical = -0.141194, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (13, 1) (val = -0.005661), analytic = -0.004715, numerical = -0.004715, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (13, 2) (val = +0.001350), analytic = +0.098715, numerical = +0.098715, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (13, 3) (val = +0.493795), analytic = -0.048457, numerical = -0.048457, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (13, 4) (val = -0.133918), analytic = +0.003070, numerical = +0.003070, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (13, 5) (val = -0.298266), analytic = +0.011807, numerical = +0.011807, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (13, 6) (val = +0.134899), analytic = -0.098726, numerical = -0.098726, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (13, 7) (val = +0.629063), analytic = -0.035955, numerical = -0.035955, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (13, 8) (val = +0.155841), analytic = +0.032961, numerical = +0.032961, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (13, 9) (val = -0.041653), analytic = +0.097331, numerical = +0.097331, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (13, 10) (val = -0.486182), analytic = +0.006689, numerical = +0.006689, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (13, 11) (val = -0.124306), analytic = -0.078308, numerical = -0.078308, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (13, 12) (val = -0.296666), analytic = +0.371164, numerical = +0.371164, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (13, 13) (val = +0.412060), analytic = -0.902398, numerical = -0.902398, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (13, 14) (val = +0.139221), analytic = +0.254762, numerical = +0.254762, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (13, 15) (val = +0.140000), analytic = +0.769367, numerical = +0.769367, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (14, 0) (val = +0.217233), analytic = +0.042112, numerical = +0.042112, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (14, 1) (val = +0.353067), analytic = +0.055574, numerical = +0.055574, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (14, 2) (val = -0.232484), analytic = +0.068229, numerical = +0.068229, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (14, 3) (val = +0.049401), analytic = -0.195974, numerical = -0.195974, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (14, 4) (val = -0.588796), analytic = +0.014069, numerical = +0.014069, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (14, 5) (val = +0.180774), analytic = -0.004168, numerical = -0.004168, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (14, 6) (val = +0.326088), analytic = -0.079291, numerical = -0.079291, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (14, 7) (val = -0.003321), analytic = -0.019798, numerical = -0.019798, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (14, 8) (val = +0.279649), analytic = -0.018180, numerical = -0.018180, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (14, 9) (val = -0.270149), analytic = -0.127308, numerical = -0.127308, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (14, 10) (val = +0.312593), analytic = +0.007904, numerical = +0.007904, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (14, 11) (val = -0.254947), analytic = +0.030539, numerical = +0.030539, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (14, 12) (val = +0.414501), analytic = +0.164411, numerical = +0.164411, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (14, 13) (val = -0.003494), analytic = -0.137596, numerical = -0.137596, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (14, 14) (val = +0.135071), analytic = -0.321765, numerical = -0.321765, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param WLSTM index (14, 15) (val = -0.304316), analytic = +0.500904, numerical = +0.500904, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param c0 index (0, 0) (val = +0.991301), analytic = -0.487489, numerical = -0.487489, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param c0 index (0, 1) (val = -0.647627), analytic = +0.553454, numerical = +0.553454, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param c0 index (0, 2) (val = +0.086428), analytic = -0.236035, numerical = -0.236035, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param c0 index (0, 3) (val = +0.871517), analytic = +0.029322, numerical = +0.029322, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param c0 index (1, 0) (val = +0.034038), analytic = -0.487086, numerical = -0.487086, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param c0 index (1, 1) (val = -0.105219), analytic = +0.647853, numerical = +0.647853, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param c0 index (1, 2) (val = -1.179923), analytic = -0.279625, numerical = -0.279625, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param c0 index (1, 3) (val = -0.149360), analytic = -0.660331, numerical = -0.660331, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param c0 index (2, 0) (val = +0.234105), analytic = +0.579373, numerical = +0.579373, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param c0 index (2, 1) (val = +0.182041), analytic = +0.359690, numerical = +0.359690, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param c0 index (2, 2) (val = +0.237782), analytic = -0.677227, numerical = -0.677227, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param c0 index (2, 3) (val = -0.117634), analytic = +0.813228, numerical = +0.813228, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param h0 index (0, 0) (val = +0.153660), analytic = +0.050054, numerical = +0.050054, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param h0 index (0, 1) (val = -0.045067), analytic = -0.070577, numerical = -0.070577, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param h0 index (0, 2) (val = -0.413007), analytic = +0.129941, numerical = +0.129941, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param h0 index (0, 3) (val = -0.387672), analytic = -0.158343, numerical = -0.158343, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param h0 index (1, 0) (val = -1.924350), analytic = +0.142972, numerical = +0.142972, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param h0 index (1, 1) (val = -0.055619), analytic = -0.268212, numerical = -0.268212, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param h0 index (1, 2) (val = -1.366686), analytic = +0.197478, numerical = +0.197478, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param h0 index (1, 3) (val = -0.372973), analytic = +0.175027, numerical = +0.175027, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param h0 index (2, 0) (val = +0.454577), analytic = -0.057464, numerical = -0.057464, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param h0 index (2, 1) (val = +0.670989), analytic = -0.109786, numerical = -0.109786, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param h0 index (2, 2) (val = -0.390939), analytic = -0.053038, numerical = -0.053038, relative error = +0.000000
n= 5 b= 3 d= 4
n= 5 b= 3 d= 4
OK checking param h0 index (2, 3) (val = +1.155490), analytic = -0.073782, numerical = -0.073782, relative error = +0.000000
every line should start with OK. Have a nice day!
(minGPT) PS D:\ccc\ai\_diy\karpathy\more>
