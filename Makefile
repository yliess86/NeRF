BLENDER=data/blender
SCENE=lego

TRAIN=python3 -m nerf.train
INFER=python3 -m nerf.infer
BENCH=python3 -m nerf.bench
DISTILL=python3 -m nerf.distill

FCONFIG=--nerf_width 256 --nerf_depth 8 --nerf_resid
TCONFIG=--nerf_width 32 --nerf_depth 2

NERF=res/NeRF
DFNERF=res/DistillFullNeRF
DTNERF=res/DistillTinyNeRF
HNERF=res/HybridNeRF

INFER_OPTS=--width 400 --height 400 --frames 50
BENCH_OPTS=--width 400 --height 400
AMP=--amp

SAMPLES=--coarse 64 --fine 64

all: train distill_full distill_tiny hybrid bench_all

train:
	mkdir -p ${NERF}
	
	${TRAIN} -i ${BLENDER} -s ${SCENE} -o res/NeRF ${FCONFIG} --perturb ${AMP} -r
	${INFER} -i ${NERF}/NeRF_${SCENE}.model.ts ${SAMPLES} ${INFER_OPTS} ${AMP}

distill_full:
	mkdir -p ${DFNERF}
	
	${DISTILL} -i ${BLENDER} -s ${SCENE} -o ${DFNERF} -t ${NERF}/NeRF_${SCENE}.model.ts ${FCONFIG} --perturb ${AMP}
	${INFER} -i ${DFNERF}/NeRF_${SCENE}.model.ts ${SAMPLES} ${INFER_OPTS} ${AMP}

distill_tiny:
	mkdir -p ${DTNERF}

	${DISTILL} -i ${BLENDER} -s ${SCENE} -o ${DTNERF} -t ${NERF}/NeRF_${SCENE}.model.ts  ${TCONFIG} --perturb ${AMP}
	${INFER} -i ${DTNERF}/NeRF_${SCENE}.model.ts ${SAMPLES} ${INFER_OPTS} ${AMP}

hybrid:
	mkdir -p ${HNERF}
	
	cp ${DFNERF}/NeRF_${SCENE}.model.ts ${HNERF}/NeRF_${SCENE}.fine.model.ts
	cp ${DTNERF}/NeRF_${SCENE}.model.ts ${HNERF}/NeRF_${SCENE}.coarse.model.ts

	${INFER} -i ${HNERF}/NeRF_${SCENE}.coarse.model.ts ${HNERF}/NeRF_${SCENE}.fine.model.ts ${SAMPLES} ${INFER_OPTS} ${AMP}

bench_all:
	${BENCH} -i ${NERF}/NeRF_${SCENE}.model.ts ${BENCH_OPTS}
	${BENCH} -i ${DFNERF}/NeRF_${SCENE}.model.ts ${BENCH_OPTS}
	${BENCH} -i ${DTNERF}/NeRF_${SCENE}.model.ts ${BENCH_OPTS}
	${BENCH} -i ${HNERF}/NeRF_${SCENE}.coarse.model.ts ${HNERF}/NeRF_${SCENE}.fine.model.ts ${BENCH_OPTS}
