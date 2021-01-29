pushd ../graph
for value in {0..9}
do
  TF_CPP_MIN_LOG_LEVEL="3" python3 -W ignore neuroGRS_graph_expt.py seed0V$value 04e1
  TF_CPP_MIN_LOG_LEVEL="3" python3 -W ignore neuroGRS_graph_expt.py seed0V$value 04e2
  TF_CPP_MIN_LOG_LEVEL="3" python3 -W ignore neuroGRS_graph_expt.py seed0V$value 04e3
  TF_CPP_MIN_LOG_LEVEL="3" python3 -W ignore neuroGRS_graph_expt.py seed0V$value 05e1
  TF_CPP_MIN_LOG_LEVEL="3" python3 -W ignore neuroGRS_graph_expt.py seed0V$value 05e2
  TF_CPP_MIN_LOG_LEVEL="3" python3 -W ignore neuroGRS_graph_expt.py seed0V$value 05e3
  TF_CPP_MIN_LOG_LEVEL="3" python3 -W ignore neuroGRS_graph_expt.py seed0V$value 06e1
  TF_CPP_MIN_LOG_LEVEL="3" python3 -W ignore neuroGRS_graph_expt.py seed0V$value 06e2
  TF_CPP_MIN_LOG_LEVEL="3" python3 -W ignore neuroGRS_graph_expt.py seed0V$value 06e3
done
popd