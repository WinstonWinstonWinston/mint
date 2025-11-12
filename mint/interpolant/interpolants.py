import torch
from .abstract import LinearInterpolant
from .corrector import Corrector, IdentityCorrector

class TemporallyLinearInterpolant(LinearInterpolant):
    r"""
    Temporally Linear interpolant :math:`I(t, x_0, x_1) = (1 - t) x_0 + t x_1` between tensors
    :math:`x_0` and :math:`x_1` from two distributions :math:`p_0` and :math:`p_1` at time :math:`t`.

    .. math::

    I(t, x_0, x_1) = (1 - t)\,x_0 + t\,x_1
    """

    def __init__(self, velocity_weight: float = 1.0, denoiser_weight: float = 1.0, gamma_weight = 1.0) -> None:
        super().__init__(velocity_weight, denoiser_weight)
        self.gamma_weight = gamma_weight
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Alpha function :math:`\alpha(t) = (1-t)`.

        :param t:
            Times in :math:`t \in [0,1]`.
        :type t: torch.Tensor

        :return:
            Values of the alpha function :math:`\alpha(t) = 1-t` at the given times :math:`t`.
        :rtype: torch.Tensor
        """
        return 1.0 - t

    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Derivative of the alpha function :math:`\dot{\alpha}(t) = -1`.

        :param t:
            Times in :math:`t \in [0,1]`.
        :type t: torch.Tensor

        :return:
            Derivatives of the alpha function :math:`\dot{\alpha}(t)` = -1 at the given times :math:`t`.
        :rtype: torch.Tensor
        """
        return -torch.ones_like(t)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Beta function :math:`\beta(t) = t`.

        :param t:
            Times in :math:`t \in [0,1]`.
        :type t: torch.Tensor

        :return:
            Values of the beta function :math:`\beta(t) = t` at the given times :math:`t`.
        :rtype: torch.Tensor
        """
        return t.clone()

    def beta_dot(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Derivative of the beta function :math:`\dot{\beta}(t) = 1`.

        :param t:
            Times in :math:`t \in [0,1]`.
        :type t: torch.Tensor

        :return:
            Derivatives of the beta function :math:`\dot{\beta}(t) = 1` at the given times :math:`t`.
        :rtype: torch.Tensor
        """
        return torch.ones_like(t)

    def get_corrector(self) -> Corrector:
        """
        :return:
            Identity corrector that does nothing.
        :rtype: Corrector
        """
        return IdentityCorrector()
    
    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Gamma function :math:`\gamma(t) = \sqrt{2t(1-t)}`.

        :param t:
            Times in :math:`t \in [0,1]`.
        :type t: torch.Tensor

        :return:
            Values of the gamma function :math:`\gamma(t)` at the given times.
        :rtype: torch.Tensor
        """
        return self.gamma_weight*torch.sqrt(2*t*(1-t))
    
    def gamma_dot(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Time derivative of the gamma function :math:`\dot{\gamma}(t) = (1-2t)\gamma(t)`.

        :param t:
            Times in :math:`t \in [0,1]`.
        :type t: torch.Tensor

        :return:
            Derivatives of the gamma function :math:`\dot{\gamma}(t) = (1-2t)/\gamma(t)` at the given times.
        :rtype: torch.Tensor
        """
        return self.gamma_weight*(1 / (2 * torch.sqrt(2 * t * (1 - t)))) * (2 * (1 - t) - 2 * t)

class TrigonometricInterpolant(LinearInterpolant):
    """
    Trigonometric interpolant I(t, x_0, x_1) = cos(pi / 2 * t) * x_0 + sin(pi / 2 * t) * x_1 between points x_0 and x_1
    from two distributions p_0 and p_1 at times t.
    """
    def __init__(self, velocity_weight: float = 1.0, denoiser_weight: float = 1.0) -> None:
        """
        Construct trigonometric interpolant.
        """
        super().__init__(velocity_weight, denoiser_weight)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """
        Alpha function alpha(t) in the linear interpolant.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Values of the alpha function at the given times.
        :rtype: torch.Tensor
        """
        return torch.cos(torch.pi * t / 2.0)

    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        """
        Time derivative of the alpha function in the linear interpolant.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Derivatives of the alpha function at the given times.
        :rtype: torch.Tensor
        """
        return -(torch.pi / 2.0) * torch.sin(torch.pi * t / 2.0)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """
        Beta function beta(t) in the linear interpolant.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Values of the beta function at the given times.
        :rtype: torch.Tensor
        """
        return torch.sin(torch.pi * t / 2.0)

    def beta_dot(self, t: torch.Tensor) -> torch.Tensor:
        """
        Time derivative of the beta function in the linear interpolant.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Derivatives of the beta function at the given times.
        :rtype: torch.Tensor
        """
        return (torch.pi / 2.0) * torch.cos(torch.pi * t / 2.0)

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Identity corrector that does nothing.
        :rtype: Corrector
        """
        return IdentityCorrector()

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Gamma function in the stochastic interpolant.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Values of the gamma function at the given times.
        :rtype: torch.Tensor
        """
        return torch.sqrt(2*t*(1-t))
    
    def gamma_dot(self, t: torch.Tensor) -> torch.Tensor:
        """
        Time derivative of the gamma function in the stochastic interpolant.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Derivatives of the gamma function at the given times.
        :rtype: torch.Tensor
        """
        return (1-2*t)/self.gamma(t)

class EncoderDecoderInterpolant(LinearInterpolant):
    """
    Encoder-decoder interpolant
    I(t, x_0, x_1) = cos^2(pi * (t - switch_time * t)^p / ((switch_time - switch_time * t)^p +  (t - switch_time * t)^p)) * 1_[0, switch_time) * x_0
                   + cos^2(pi * (t - switch_time * t)^p / ((switch_time - switch_time * t)^p + (t - switch_time * t)^p)) * 1_(1-switch_time, 1] * x_1
    between points x_0 and x_1 from two distributions p_0 and p_1 at times t.

    gamma(t) = sqrt(a) * sin^2(pi * (t - switch_time * t)^p / ((switch_time - switch_time * t)^p + (t - switch_time * t)^p))

    For a=1 p=1 and switch_time=0.5, this interpolant becomes
    I(t, x_0, x_1) = cos^2(pi * t) * 1_[0, switch_time) * x_0 + cos^2(pi * t) * 1_(1-switch_time, 1] * x_1,
    and gamma(t) = sin^2(pi * t) which was considered in the stochastic interpolants paper.

    Note that the time derivatives are only bounded for p>=0.5.

    :param a:
        Constant a > 0.
        Defaults to 1.0.
    :type a: float
    :param switch_time:
        Time in (0, 1) at which to switch from x_0 to x_1.
        Defaults to 0.5.
    :type switch_time: float
    :param power:
        Power p in the interpolant.
        Defaults to 1.0.
    :type power: float
    :param velocity_weight:
        Constant velocity_weight > 0 which scaless loss of the velocity
    :type velocity_weight: float
    :param denoiser_weight:
        Constant denoiser_weight > 0 which scaless loss of the denoiser
    :type velocity_weight: float
    

    :raises ValueError:
        If switch_time is not in (0,1) or power is less than 0.5.
    """
    def __init__(self, a: float = 1.0, switch_time: float = 0.5, power: float = 1.0, 
                 velocity_weight: float = 1.0, denoiser_weight: float = 1.0) -> None:
        """Construct encoder-decoder interpolant."""
        super().__init__(velocity_weight, denoiser_weight)
        if a <= 0.0:
            raise ValueError("Constant a must be positive.")
        if switch_time <= 0.0 or switch_time >= 1.0:
            raise ValueError("Switch time must be in (0,1).")
        if power < 0.5:
            raise ValueError("Power must be at least 0.5.")
        self._sqrt_a = a ** 0.5
        self._switch_time = switch_time
        self._power = power
        

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """
        Alpha function alpha(t) in the linear interpolant.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Values of the alpha function at the given times.
        :rtype: torch.Tensor
        """
        a = (t - self._switch_time * t) ** self._power
        b = (self._switch_time - self._switch_time * t) ** self._power + a
        return torch.where(t <= self._switch_time, torch.cos(torch.pi * a / b) ** 2, 0.0)

    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        """
        Time derivative of the alpha function in the linear interpolant.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Derivatives of the alpha function at the given times.
        :rtype: torch.Tensor
        """
        # In principle, one should be careful with floating point precision here as t->0 and t->1, especially for
        # self._power=1/2. However, time does not get arbitrarily close to 0 or 1 in practice. We assert this here so
        # this gets updated if omg.globals.SMALL_TIME or omg.globals.BIG_TIME change.
        assert torch.all((1.0e-3 <= t) & (t <= 1.0 - 1.0e-3))
        a = (t - self._switch_time * t) ** self._power
        b = (self._switch_time - self._switch_time * t) ** self._power
        c = torch.sin(2.0 * torch.pi * a / (a + b))
        return torch.where(
            t <= self._switch_time,
            self._power * torch.pi * a * b * c / (t * (t - 1.0) * ((a + b) ** 2)),
            0.0)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """
        Beta function beta(t) in the linear interpolant.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Values of the beta function at the given times.
        :rtype: torch.Tensor
        """
        a = (t - self._switch_time * t) ** self._power
        b = (self._switch_time - self._switch_time * t) ** self._power + a
        return torch.where(t > self._switch_time, torch.cos(torch.pi * a / b) ** 2, 0.0)

    def beta_dot(self, t: torch.Tensor) -> torch.Tensor:
        """
        Time derivative of the beta function in the linear interpolant.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Derivatives of the beta function at the given times.
        :rtype: torch.Tensor
        """
        # In principle, one should be careful with floating point precision here as t->0 and t->1, especially for
        # self._power=1/2. However, time does not get arbitrarily close to 0 or 1 in practice. We assert this here so
        # this gets updated if omg.globals.SMALL_TIME or omg.globals.BIG_TIME change.
        assert torch.all((1.0e-3 <= t) & (t <= 1.0 - 1.0e-3))
        a = (t - self._switch_time * t) ** self._power
        b = (self._switch_time - self._switch_time * t) ** self._power
        c = torch.sin(2.0 * torch.pi * a / (a + b))
        return torch.where(
            t > self._switch_time,
            self._power * torch.pi * a * b * c / (t * (t - 1.0) * ((a + b) ** 2)),
            0.0)

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Identity corrector that does nothing.
        :rtype: Corrector
        """
        return IdentityCorrector()
    
    def gamma(self, t: torch.Tensor) -> torch.Tensor:
            """
            Evaluate the gamma function gamma(t) in the latent variable gamma(t) * z at the times t.

            :param t:
                Times in [0,1].
            :type t: torch.Tensor

            :return:
                Gamma function gamma(t).
            :rtype: torch.Tensor
            """
            a = (t - self._switch_time * t) ** self._power
            b = (self._switch_time - self._switch_time * t) ** self._power + a
            return self._sqrt_a * torch.sin(torch.pi * a / b) ** 2

    def gamma_derivative(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative of the gamma function gamma(t) in the latent variable gamma(t) * z with respect to time.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Derivative of the gamma function.
        :rtype: torch.Tensor
        """
        # In principle, one should be careful with floating point precision here as t->0 and t->1, especially for
        # self._power=1/2. However, time does not get arbitrarily close to 0 or 1 in practice. We assert this here so
        # this gets updated if omg.globals.SMALL_TIME or omg.globals.BIG_TIME change.
        assert torch.all((1.0e-3 <= t) & (t <= 1.0 - 1.0e-3))
        a = (t - self._switch_time * t) ** self._power
        b = (self._switch_time - self._switch_time * t) ** self._power
        c = torch.sin(2.0 * torch.pi * a / (a + b))
        return -self._sqrt_a * self._power * torch.pi * a * b * c / (t * (t - 1.0) * ((a + b) ** 2))
    
class MirrorInterpolant(LinearInterpolant):
    """
    Mirror interpolant I(t, x_0, x_1) = x_1 between points x_0 and x_1 from the same distribution p_1 at times t.

    :param velocity_weight:
        Constant velocity_weight > 0 which scaless loss of the velocity
    :type velocity_weight: float
    :param denoiser_weight:
        Constant denoiser_weight > 0 which scaless loss of the denoiser
    :type velocity_weight: float
    """
    def __init__(self, velocity_weight: float = 1.0, denoiser_weight: float = 1.0) -> None:
        """
        Construct mirror interpolant.
        """
        super().__init__(velocity_weight,denoiser_weight)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """
        Alpha function alpha(t) in the mirror interpolant.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Values of the alpha function at the given times.
        :rtype: torch.Tensor
        """
        return torch.zeros_like(t)

    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        """
        Time derivative of the alpha function in the mirror interpolant.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Derivatives of the alpha function at the given times.
        :rtype: torch.Tensor
        """
        return torch.zeros_like(t)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """
        Beta function beta(t) in the mirror interpolant.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Values of the beta function at the given times.
        :rtype: torch.Tensor
        """
        return torch.ones_like(t)

    def beta_dot(self, t: torch.Tensor) -> torch.Tensor:
        """
        Time derivative of the beta function in the mirror interpolant.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Derivatives of the beta function at the given times.
        :rtype: torch.Tensor
        """
        return torch.zeros_like(t)
    
    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Gamma function in the stochastic interpolant.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Values of the gamma function at the given times.
        :rtype: torch.Tensor
        """
        return torch.sqrt(2*t*(1-t))
    
    def gamma_dot(self, t: torch.Tensor) -> torch.Tensor:
        """
        Time derivative of the gamma function in the stochastic interpolant.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Derivatives of the gamma function at the given times.
        :rtype: torch.Tensor
        """
        return (1-2*t)/self.gamma(t)

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Identity corrector that does nothing.
        :rtype: Corrector
        """
        return IdentityCorrector()