pushd ../graph
for value in {0..9}
do
  TF_CPP_MIN_LOG_LEVEL="3" python3 -W ignore neuroGRS_graph.py 1 V{$value}seed0 04e1
  TF_CPP_MIN_LOG_LEVEL="3" python3 -W ignore neuroGRS_graph.py 1 V{$value}seed0 04e2
  TF_CPP_MIN_LOG_LEVEL="3" python3 -W ignore neuroGRS_graph.py 1 V{$value}seed0 04e3
  TF_CPP_MIN_LOG_LEVEL="3" python3 -W ignore neuroGRS_graph.py 1 V{$value}seed0 05e1
  TF_CPP_MIN_LOG_LEVEL="3" python3 -W ignore neuroGRS_graph.py 1 V{$value}seed0 05e2
  TF_CPP_MIN_LOG_LEVEL="3" python3 -W ignore neuroGRS_graph.py 1 V{$value}seed0 05e3
  TF_CPP_MIN_LOG_LEVEL="3" python3 -W ignore neuroGRS_graph.py 1 V{$value}seed0 06e1
  TF_CPP_MIN_LOG_LEVEL="3" python3 -W ignore neuroGRS_graph.py 1 V{$value}seed0 06e2
  TF_CPP_MIN_LOG_LEVEL="3" python3 -W ignore neuroGRS_graph.py 1 V{$value}seed0 06e3
done
popd