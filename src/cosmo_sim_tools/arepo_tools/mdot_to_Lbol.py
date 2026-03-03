def get_conversion_factor_mb2(radiative_efficiency):
	c=3e8
	mass_unit_conv=1e10/980/1e6
	mass_sun=2e30
	yr_to_sec=3.15e7
	lamb=radiative_efficiency
	h=0.7
	joule_to_ergs=1e7
	total_conv=mass_unit_conv*mass_sun/yr_to_sec*c**2*joule_to_ergs*lamb
	return total_conv


def get_conversion_factor_arepo(radiative_efficiency):
	c=3e8
	mass_unit_conv=1e10/978/1e6
	mass_sun=2e30
	yr_to_sec=3.15e7
	lamb=radiative_efficiency
	joule_to_ergs=1e7
	total_conv=mass_unit_conv*mass_sun/yr_to_sec*c**2*joule_to_ergs*lamb
	return total_conv
