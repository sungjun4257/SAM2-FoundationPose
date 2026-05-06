# SAM2-FoundationPose

[NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose) 를 기반으로
**SAM2 segmentation**, **RealSense 카메라**, **ZMQ 서빙**, **GraspNet 연동** 기능을 위한 repo.

---

## 설치

설치 과정은 **원본 저장소 README** 참고. 

https://github.com/NVlabs/FoundationPose
https://github.com/IDEA-Research/Grounded-SAM-2

추가 의존성:
- realsense 카메라 lib 설치
---

## 주로 쓰는 명령어

```bash
python run_demo_cad_with_sam2.py
```

CAD 모델 + SAM2 자동 segmentation 으로 6D pose 추정.

---

<!-- ## 그 외 스크립트

| 스크립트 | 용도 |
|---|---|
| `run_demo_cad_with_sam2.py` | **메인** — CAD + SAM2 |
| `run_demo_cad_with_sam2_one_class.py` | 단일 클래스만 트래킹 |
| `run_demo_cad_with_realsense.py` | RealSense 카메라 실시간 입력 |
| `run_demo_cad_with_scale_balanced_grasp.py` | Scale-balanced grasp 연동 |
| `run_demo_realsense.py` | RealSense (CAD 없이) |
| `run_demo_cad.py` | CAD 기반 커스텀 데모 |
| `run_demo_zmq.py` / `run_demo_zmq_with_sam2.py` | ZMQ 서버로 pose 서빙 |
| `zmq_foundationpose.py` | ZMQ 클라이언트/래퍼 |
| `run_graspnet_evaluate.py` | GraspNet 데이터셋 평가 |
| `interactive.py`, `model_analysis.py`, `obj_params.py` | 디버깅/분석 유틸 |

원본 데모 (`run_demo.py`, `run_linemod.py`, `run_ycb_video.py`) 도 그대로 사용 가능.

--- -->

## Citation

원본 논문 인용:
```bibtex
@InProceedings{foundationposewen2024,
  author    = {Bowen Wen and Wei Yang and Jan Kautz and Stan Birchfield},
  title     = {{FoundationPose}: Unified 6D Pose Estimation and Tracking of Novel Objects},
  booktitle = {CVPR},
  year      = {2024},
}
```

## License

원본 NVIDIA Source Code License 를 따릅니다. Copyright © 2024, NVIDIA Corporation.
