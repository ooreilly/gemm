arch=sm_70
flags=-lcublas --ptxas-options=-v -lineinfo
compile:
	nvcc -arch=$(arch) -use_fast_math $(flags) main.cu -o gemm

profile: compile
	nv-nsight-cu-cli --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,smsp__inst_executed_op_global_ld.sum,smsp__inst_executed_op_global_st.sum \
	./gemm 8192 8192 8192 > nv-nsight-perrformance-results.txt

