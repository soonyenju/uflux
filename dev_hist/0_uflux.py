from scieco import photosynthesis, respiration, evapotranspiration

def get_lue_iwue_grid(nct):
    nct = nct.copy()
    tc =  nct['temperature_2m']
    tc = tc.where(tc >= -30, -30)
    tc = tc.values
    co2 = nct['co2'].values
    patm = nct['surface_pressure'].values * 100
    vpd = nct['VPD'].values * 100

    ca = photosynthesis.calc_co2_to_ca(co2, patm)
    c3_lue, iwue = photosynthesis.calc_light_water_use_efficiency(tc, patm, ca, vpd, True, c4 = False, limitation_factors = 'wang17')
    c4_lue, _ = photosynthesis.calc_light_water_use_efficiency(tc, patm, ca, vpd, True, c4 = True, limitation_factors = 'none')

    c3_lue = xr.DataArray(c3_lue, dims = ["time", "latitude", "longitude"], coords={"time": nct.time, "latitude": nct.latitude, "longitude": nct.longitude}).rename('C3_LUE')
    c4_lue = xr.DataArray(c4_lue, dims = ["time", "latitude", "longitude"], coords={"time": nct.time, "latitude": nct.latitude, "longitude": nct.longitude}).rename('C4_LUE')
    iwue = xr.DataArray(iwue, dims = ["time", "latitude", "longitude"], coords={"time": nct.time, "latitude": nct.latitude, "longitude": nct.longitude}).rename('IWUE')
    lue = (c3_lue * (1 - nct['C4_area'].fillna(0) / 100) + c4_lue  * nct['C4_area'].fillna(0) / 100)
    return lue, iwue