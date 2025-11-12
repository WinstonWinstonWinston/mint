from abc import ABC, abstractmethod
import torch
from typing import Dict


class Corrector(ABC):
    r"""
    Abstract class for defining a corrector function that corrects the input :math:`x` (for instance, wrapping back coordinates
    to a specific cell in periodic boundary conditions).
    """

    @abstractmethod
    def correct(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Correct the input :math:`x`.

        :param x:
            Input to correct.
        :type x: torch.Tensor

        :return:
            Corrected input.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def unwrap(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        r"""
        Correct the input :math:`x_1` based on the reference input :math:`x_0` (for instance, return the image of :math:`x_1` closest to  :math:`x_0` in
        periodic boundary conditions).

        :param x_0:
            Reference input.
        :type x_0: torch.Tensor
        :param x_1:
            Input to correct.
        :type x_1: torch.Tensor

        :return:
            Unwrapped x_1 value.
        :rtype: torch.Tensor
        """
        raise NotImplementedError
    

class Interpolant(ABC):
    r"""
    Abstract class for defining an interpolant

    .. math::

        x_t = I(t, x_0, x_1) + \gamma(t)z
    
    in a stochastic interpolant between points  :math:`x_0` and  :math:`x_1` from two distributions  :math:`p_0` and  :math:`p_1` at times t.

    :param velocity_weight:
        Constant velocity_weight > 0 which scaless loss of the velocity
    :type velocity_weight: float
    :param denoiser_weight:
        Constant denoiser_weight > 0 which scaless loss of the denoiser
    :type velocity_weight: float
    """
    def __init__(self, velocity_weight: float = 1.0, denoiser_weight: float = 1.0) -> None:
        self.velocity_weight = velocity_weight
        self.denoiser_weight = denoiser_weight

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        r"""
        Interpolate between points :math:`x_0` and :math:`x_1` from two distributions :math:`p_0` and :math:`p_1` at times :math:`t`.

        In order to possibly allow for periodic boundary conditions, :math:`x_1` is first unwrapped based on the corrector of
        this interpolant. For the identity corrector, this unwrapping does nothing. For periodic boundary conditions,
        this unwrapping returns the closest image of :math:`x_1` to :math:`x_0`. The interpolant is then computed based on the unwrapped
        :math:`x_1` and function :math:`I(t, x_0, x_1)`

        :param t:
            Times in :math:`t \in [0,1]`.
        :type t: torch.Tensor
        :param x_0:
            Points sampled from :math:`p_0`.
        :type x_0: torch.Tensor
        :param x_1:
            Points sampled from :math:`p_1`.
        :type x_1: torch.Tensor

        :return:
            Interpolated value :math:`x_t` and the latent noise :math:`z`.
        :rtype:  tuple[torch.Tensor, torch.Tensor]
        """
        raise NotImplementedError

    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the derivative of the interpolant :math:`\dot{x}_t` with respect to time between points :math:`x_0` and :math:`x_1` 
        from two distributions :math:`p_0` and :math:`p_1` at times :math:`t`.

        :param t:
            Times in :math:`t \in [0,1]`.
        :type t: torch.Tensor
        :param x_0:
            Points sampled from :math:`p_0`.
        :type x_0: torch.Tensor
        :param x_1:
            Points sampled from :math:`p_1`.
        :type x_1: torch.Tensor
        :param z:
            Latent normally distributed noise :math:`z \sim N(0,1)`.
        :type z: torch.Tensor

        :return:
            Derivative of the interpolant.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def get_corrector(self) -> Corrector:
        r"""
        Get the corrector implied by the interpolant (for instance, a corrector that considers periodic boundary
        conditions).

        :return:
            Corrector.
        :rtype: Corrector
        """
        raise NotImplementedError
    
    @abstractmethod
    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Gamma function :math:`\gamma(t)` in the stochastic interpolant.

        :param t:
            Times in :math:`t \in [0,1]`.
        :type t: torch.Tensor

        :return:
            Values of the gamma function :math:`\gamma(t)` at the given times.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def gamma_dot(self, t: torch.Tensor):
        r"""
        Time derivative of the gamma function :math:`\dot{\gamma}(t)` in the stochastic interpolant.

        :param t:
            Times in :math:`t \in [0,1]`.
        :type t: torch.Tensor

        :return:
            Derivatives of the gamma function :math:`\dot{\gamma}(t)` at the given times.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def loss(self, t, x_0, x_1, z, b, eta=None) -> Dict[str, torch.Tensor]:
        r"""
        Loss value for a batch of data. If the eta term is None this corresponds only to the velocity loss.
        Otherwise it gives a weighted average between them based off of init params velocity_weight, and denoiser_weight.

        * :math:`\mathcal{L}_{\text{velocity}}(\theta) = \mathbb{E}\!\left[\|b\|^{2} - 2\, b \cdot \dot I\right]`
        * :math:`\mathcal{L}_{\text{denoiser}}(\theta) = \mathbb{E}\!\left[\|\eta\|^{2} - 2\, \eta \cdot z\right]`
        * :math:`(w_\eta + w_b)\mathcal{L}(\theta) = w_b\,\mathcal{L}_{\text{velocity}}(\theta) + w_\eta\,\mathcal{L}_{\text{denoiser}}(\theta)`

        :param t:
            Times in :math:`t \in [0,1]`.
        :type t: torch.Tensor
        :param x_0:
            Points sampled from :math:`p_0`.
        :type x_0: torch.Tensor
        :param x_0:
            Points sampled from :math:`p_1`.
        :type x_0: torch.Tensor
        :param z:
            Latent normally distributed noise :math:`z \sim N(0,1)`.
        :type z: torch.Tensor
        :param b:
            Predicted velocity values :math:`b(t)` for :math:`x_t`.
        :type eta: torch.Tensor
        :param eta:
            Predicted denoiser values :math:`\eta(t)` for :math:`x_t`.
        :type eta: torch.Tensor

        :return:
            A dictionary of loss values with keys loss, loss_velocity, and loss_denoiser
        :rtype: dict[str, torch.Tensor]
        """
        interpolant_dot = self.interpolate_derivative(t,x_0,x_1,z)
        loss_velocity  = torch.mean(torch.einsum('BD,BD->B', b, b    ) - 2*torch.einsum('BD, BD', b, interpolant_dot))
        loss_denoiser  = torch.mean(torch.einsum('BD,BD->B', eta, eta) - 2*torch.einsum('BD, BD', eta, z) if eta is not None else torch.zeros_like(loss_velocity))
        loss = (self.velocity_weight*loss_velocity + self.denoiser_weight*loss_denoiser)/(self.velocity_weight + self.denoiser_weight)
        return {"loss": loss,
                "loss_velocity": loss_velocity,
                "loss_denoiser": loss_denoiser}

    def step_sde(self, x_t:torch.Tensor, b:torch.Tensor, eta:torch.Tensor, t:torch.Tensor, dt:float, epsilon) -> torch.Tensor:
        r"""One Euler-Maruyama step for an SDE with time-dependent noise.

        Continuous form:

        .. math::

            dX_t = \Big[b(X_t,t) - \tfrac{\epsilon(t)\,\eta (X_t,t)}{\gamma(t)}\Big]\,dt + \sqrt{2\,\epsilon(t)}\,dW_t

        Discrete update:

        .. math::

            X_{t+\Delta t} = x_t + \Big[b(X_t,t) - \tfrac{\epsilon(t)\,\eta (X_t,t)}{\gamma(t)}\Big]\Delta t + \sqrt{2\,\epsilon(t)\,\Delta t}\;\xi,\quad
        
        where :math:`\xi \sim \mathcal N(0,I)`.

        :param x_t: State at time :math:`t`.
        :type x_t: torch.Tensor
        :param b: Drift term.
        :type b: torch.Tensor | float
        :param eta: Coefficient in the drift correction.
        :type eta: torch.Tensor | float
        :param t: Current time (used in :math:`\epsilon(t), \gamma(t)`).
        :type t: torch.Tensor | float
        :param dt: Time step :math:`\Delta t > 0`.
        :type dt: float
        :param epsilon: Noise schedule :math:`\epsilon(t)`.
        :type epsilon: Callable[[torch.Tensor | float], torch.Tensor | float]
        :returns: Next state :math:`X_{t+\Delta t}` (same shape/device as ``x_t``).
        :rtype: torch.Tensor
        """
        return (
            x_t                                                     # Previous state
            + (b - epsilon(t) * eta / self.gamma(t)) * dt           # drift
            + (2 * epsilon(t) * dt) ** 0.5 * torch.randn_like(x_t)  # volatility
        )

    def step_ode(self, x_t: torch.Tensor, b:torch.Tensor, dt:float) -> torch.Tensor:
        r"""One forward Euler step for an ODE.

        Continuous form:

        .. math::

            \tfrac{d}{dt}X_t = b(X_t,t)

        Discrete update:

        .. math::

            X_{t+\Delta t} = X_t + b(X_t,t)\,\Delta t

        :param x_t: State at time :math:`t`.
        :type x_t: torch.Tensor
        :param b: Drift/velocity evaluated at :math:`(x_t, t)`.
        :type b: torch.Tensor | float
        :param dt: Time step :math:`\Delta t > 0`.
        :type dt: float
        :returns: Next state :math:`x_{t+\Delta t}` (same shape/device as ``x_t``).
        :rtype: torch.Tensor
        """
        return x_t + b*dt


    def integrate(
        self,
        batch: Dict[str, torch.Tensor],
        model,
        dt: float,
        step_type: str = "sde",
        clip_val: float = 1e-3,
        epsilon=lambda t: t,
    ) -> Dict[str, torch.Tensor]:
        r"""Integrate the (S)DE forward and save the trajectory of :math:`x_t`.

        For the ODE case (forward Euler):

        .. math::

            X_{t+\Delta t} = X_t + b(X_t,t)\,\Delta t.

        For the SDE case (Euler-Maruyama):

        .. math::

            X_{t+\Delta t} = X_t + \Big[b(X_t,t) - \tfrac{\epsilon(t)\,\eta (X_t,t)}{\gamma(t)}\Big]\Delta t + \sqrt{2\,\epsilon(t)\,\Delta t}\;\xi,\quad
    
        where :math:`\xi \sim \mathcal N(0,I)`.

        :param batch: Mini-batch dict with at least ``'x'`` (current state).
        :type batch: Dict[str, torch.Tensor]
        :param model: Callable that maps/updates ``batch`` and provides fields like
                    ``'b'`` (and optionally ``'eta'``).
        :param dt: Time step :math:`\Delta t > 0`.
        :type dt: float
        :param step_type: ``"ode"`` (deterministic) or ``"sde"`` (stochastic).
        :type step_type: str
        :param clip_val: Start at :math:`t=\text{clip_val}` and end at
                        :math:`t=1-\text{clip_val}` (exclusive of the end).
        :type clip_val: float
        :param epsilon: Noise schedule :math:`\epsilon(t)`.
        :type epsilon: Callable

        :returns: A dict with:
                - ``'x_traj'``: stacked trajectory of shape
                    ``(num_steps+1, batch, ...)`` including the initial state.
                - ``'t_grid'``: times used (shape ``(num_steps,)``).
                - ``'x'``: final state (same as ``x_traj[-1]``).
        :rtype: Dict[str, torch.Tensor]
        """
        if dt <= 0:
            raise ValueError("dt must be positive.")
        
        B = int(torch.max(batch['batch']) +1)

        # Build time grid on the same device/dtype as x for consistency.
        x_t = batch["x"]
        t_grid = torch.arange(
            clip_val,
            1 - clip_val,
            dt,
            device=x_t.device,
            dtype=torch.float32,
        )

        # Choose the stepping function.
        is_ode = (step_type.lower() == "ode")

        # Store trajectory (include initial state before any step).
        x_traj = [x_t.clone()]

        # Integrate forward in time.
        for t_val in t_grid:
            # Broadcast t to batch length
            t = t_val.repeat(B)

            # Run the model to compute drift for the current batch/state.
            batch = model(batch)

            # Drift term is required.
            b = batch["b"]

            # Eta is optional; if missing, default to b.
            eta = batch.get("eta", b)

            # Step according to the selected scheme.
            if is_ode:
                x_t = self.step_ode(x_t, b, dt)
            else:
                x_t = self.step_sde(x_t, b, eta, t, dt, epsilon)

            # Update batch and record the new state.
            batch["x"] = x_t
            x_traj.append(x_t.clone())

        # Stack into a single tensor: (num_steps+1, batch, ...)
        x_traj = torch.stack(x_traj, dim=0)

        return {
            "x_traj": x_traj,
            "t_grid": t_grid,
            "x": x_traj[-1], # final state
        }

class LinearInterpolant(Interpolant):
    r"""
    Abstract class for defining a spatially linear interpolant

    .. math::

        I(t, x_0, x_1) = \alpha(t)\cdot x_0 + \beta(t) \cdot x_1
     
    in a stochastic interpolant between points :math:`x_0` and :math:`x_1` from two distributions 
    :math:`p_0` and :math:`p_1` at times :math:`t`.
    """

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Interpolate between points x_0 and x_1 from two distributions p_0 and p_1 at times t.

        .. math::
            
            x_t = \alpha(t)\cdot x_0 + \beta(t) \cdot x_1 + \gamma(y)\cdot z

        :param t:
            Times in :math:`t \in [0,1]`.
        :type t: torch.Tensor
        :param x_0:
            Points sampled from :math:`p_0`.
        :type x_0: torch.Tensor
        :param x_1:
            Points sampled from :math:`p_1`.
        :type x_1: torch.Tensor

        :return:
            Interpolated value :math:`x_t` and the latent noise :math:`z`.
        :rtype:  tuple[torch.Tensor, torch.Tensor]
        """
        z = torch.randn_like(x_0)
        x_0prime = self.get_corrector().correct(x_0)
        x_1prime = self.get_corrector().unwrap(x_0prime, x_1)
        x_t = self.alpha(t) * x_0prime + self.beta(t) * x_1prime + z*self.gamma(t)
        return self.get_corrector().correct(x_t), z

    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        r"""
         Compute the derivative of the interpolant :math:`\dot{x}_t` with respect to time between points :math:`x_0` and :math:`x_1` 
        from two distributions :math:`p_0` and :math:`p_1` at times :math:`t`.

        .. math::
        
            \dot{x_t} = \dot{\alpha}(t)\cdot x_0 + \dot{\beta}(t)\cdot x_1 + \dot{\gamma}(y)\cdot z

        :param t:
            Times in :math:`t \in [0,1]`.
        :type t: torch.Tensor
        :param x_0:
            Points sampled from :math:`p_0`.
        :type x_0: torch.Tensor
        :param x_1:
            Points sampled from :math:`p_1`.
        :type x_1: torch.Tensor
        :param z:
            Latent normally distributed noise :math:`z \sim N(0,1)`.
        :type z: torch.Tensor

        :return:
            Derivative of the interpolant \dot{x_t}.
        :rtype: torch.Tensor
        """
        x_0prime = self.get_corrector().correct(x_0)
        x_1prime = self.get_corrector().unwrap(x_0prime, x_1)
        return self.alpha_dot(t) * x_0prime + self.beta_dot(t) * x_1prime + self.gamma_dot(t) * z

    @abstractmethod
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Alpha function :math:`\alpha(t)` in the linear interpolant.

        :param t:
            Times in :math:`t \in [0,1]`.
        :type t: torch.Tensor

        :return:
            Values of the alpha function :math:`\alpha(t)` at the given times :math:`t`.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Derivative of the alpha function :math:`\dot{\alpha}(t)` in the linear interpolant.

        :param t:
            Times in :math:`t \in [0,1]`.
        :type t: torch.Tensor

        :return:
            Derivatives of the alpha function :math:`\dot{\alpha}(t)` at the given times :math:`t`.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Beta function :math:`\beta(t)` in the linear interpolant.

        :param t:
            Times in :math:`t \in [0,1]`.
        :type t: torch.Tensor

        :return:
            Values of the beta function :math:`\beta(t)` at the given times :math:`t`.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def beta_dot(self, t: torch.Tensor):
        r"""
        Derivative of the beta function :math:`\dot{\beta}(t)` in the linear interpolant.

        :param t:
            Times in :math:`t \in [0,1]`.
        :type t: torch.Tensor

        :return:
            Derivatives of the beta function :math:`\dot{\beta}(t)` at the given times :math:`t`.
        :rtype: torch.Tensor
        """
        raise NotImplementedError