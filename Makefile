BLENDER=data/blender
SCENE=lego

TRAIN=python3 -m nerf.train
INFER=python3 -m nerf.infer
BENCH=python3 -m nerf.bench
DISTILL=python3 -m nerf.distill

NERF=res/NeRF
DNERF=res/DistillNeRF

INFER_OPTS=--width 400 --height 400 --frames 50
BENCH_OPTS=--width 400 --height 400
AMP=--amp

all: train distill

train:
	${TRAIN} -i ${BLENDER} -s ${SCENE} -o res/NeRF --nerf_resid --perturb ${AMP} -r
	${INFER} -i ${NERF}/NeRF_${SCENE}.model.ts ${INFER_OPTS} ${AMP}
	${BENCH} -i ${NERF}/NeRF_${SCENE}.model.ts ${BENCH_OPTS}

distill:
	${DISTILL} -i ${BLENDER} -s ${SCENE} -o ${DNERF} -t ${NERF}/NeRF_${SCENE}.model.ts --perturb ${AMP}
	${INFER} -i ${DNERF}/NeRF_${SCENE}.model.ts ${INFER_OPTS} ${AMP}
	${BENCH} -i ${DNERF}/NeRF_${SCENE}.model.ts ${BENCH_OPTS}