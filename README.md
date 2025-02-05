# SAC를 이용한 Walker2DBulletEnv 강화학습 프로젝트

이 프로젝트는 PyBullet의 Walker2DBulletEnv 환경에서 Soft Actor-Critic(SAC) 알고리즘을 이용해 continuous action space 문제를 해결하는 예제입니다.

## 주요 기능
- PyBullet 환경을 Gym 인터페이스로 래핑
- SAC 알고리즘 (자동 엔트로피 튜닝 포함) 구현
- 경험 리플레이 버퍼를 통한 오프폴리시 학습
- 학습 결과(에피소드별 보상)를 그래프로 시각화

## 설치 방법
1. [Anaconda](https://www.anaconda.com/products/distribution) 설치 후 가상환경 생성 및 활성화  
   ```bash
   conda create -n rl_env python=3.9
   conda activate rl_env
2. 프로젝트 폴더로 이동한 후, 필요한 패키지 설치
    ```bash
    pip install -r requirements.txt
3. VS Code에서 해당 폴더를 열고 작업
    ```bash
    python src/main.py
