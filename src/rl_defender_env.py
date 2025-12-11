import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RLDatasetDefenderEnv(gym.Env):
    """
    Entorno RL para un defensor que decide PERMIT/BLOCK sobre muestras etiquetadas.

    - Obs: vector de características de la muestra actual (X[i]).
    - Acción:
        0 = PERMIT  (dejar pasar el tráfico)
        1 = BLOCK   (bloquear el tráfico)
    - Etiqueta real (y[i]):
        benign_label -> tráfico normal
        attack_label -> tráfico malicioso

    reward_config (dict):
        tp: recompensa cuando la etiqueta es ataque y la acción es BLOCK  (true positive)
        tn: recompensa cuando la etiqueta es normal y la acción es PERMIT (true negative)
        fp: penalización cuando la etiqueta es normal y la acción es BLOCK (false positive)
        fn: penalización cuando la etiqueta es ataque y la acción es PERMIT (false negative)
        omission: término adicional cuando la acción es PERMIT (coste/bonus por no bloquear)

    Ejemplo de reward_config:
        {
            "tp": 1.0,
            "tn": 0.2,
            "fp": -1.0,
            "fn": -5.0,
            "omission": 0.5,
        }
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        benign_label: int = 0,
        attack_label: int = 1,
        reward_config: dict | None = None,
        max_steps_per_episode: int | None = None,
        shuffle: bool = True,
    ) -> None:
        super().__init__()

        # Validaciones básicas
        if X.ndim != 2:
            raise ValueError(f"X debe tener shape (n_samples, n_features), recibido {X.shape}")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y debe ser 1D y tener el mismo número de muestras que X")

        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.n_samples, self.n_features = self.X.shape

        self.benign_label = int(benign_label)
        self.attack_label = int(attack_label)

        # Config de recompensa por defecto
        default_reward_config: dict[str, float] = {
            "tp": 1.0,    # ataque bloqueado
            "tn": 0.2,    # normal permitido
            "fp": -1.0,   # normal bloqueado (FP)
            "fn": -5.0,   # ataque permitido (FN)
            "omission": 0.0,  # término adicional cuando PERMIT
        }
        reward_config = reward_config or {}
        self.reward_config: dict[str, float] = {**default_reward_config, **reward_config}

        self.shuffle = bool(shuffle)
        self.max_steps_per_episode = max_steps_per_episode or self.n_samples

        # Espacios Gymnasium
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)  # 0=PERMIT, 1=BLOCK

        # Estado interno
        self.current_idx: int = 0
        self.steps: int = 0
        self.indices = np.arange(self.n_samples, dtype=np.int64)

    # --------------------
    # API Gymnasium
    # --------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        if self.shuffle:
            self.np_random.shuffle(self.indices)

        self.current_idx = 0
        self.steps = 0

        obs = self._get_observation()
        info: dict = {}
        return obs, info

    def _get_observation(self) -> np.ndarray:
        idx = self.indices[self.current_idx]
        return self.X[idx]

    def _compute_reward(self, label: int, action: int) -> float:
        """
        Calcula la recompensa en función de la etiqueta real, la acción
        y la configuración de recompensa.

        Convención:
            0 = PERMIT, 1 = BLOCK
        """
        rc = self.reward_config

        is_attack = (label == self.attack_label)
        is_benign = (label == self.benign_label)

        reward = 0.0

        if is_attack:
            if action == 1:
                reward = rc["tp"]   # ataque bloqueado (TP)
            else:
                reward = rc["fn"]   # ataque permitido (FN)
        elif is_benign:
            if action == 0:
                # Permitir benigno = omisión: recompensa parcial
                # (omission_reward controla cuánto premio se lleva)
                reward = rc["omission"]
            else:
                reward = rc["fp"]   # normal bloqueado (FP)
        else:
            # Para etiquetas desconocidas, tratamos como benignas por defecto
            if action == 0:
                reward = rc["omission"]
            else:
                reward = rc["fp"]

        return float(reward)


    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Acción inválida: {action}")

        idx = self.indices[self.current_idx]
        label = self.y[idx]

        reward = self._compute_reward(int(label), int(action))

        # Avanzar
        self.current_idx += 1
        self.steps += 1

        terminated = (self.current_idx >= self.n_samples)
        truncated = (self.steps >= self.max_steps_per_episode)

        if not (terminated or truncated):
            obs = self._get_observation()
        else:
            obs = self.X[idx]

        info = {
            "sample_index": int(idx),
            "true_label": int(label),
        }

        return obs, reward, bool(terminated), bool(truncated), info

    def render(self):
        # No necesitamos render para este entorno (tabular)
        pass

    def close(self):
        pass
