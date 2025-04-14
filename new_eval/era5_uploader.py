import cdsapi
import xarray as xr
import cfgrib
import os
from dotenv import load_dotenv
from shutil import rmtree
import pdb
from huggingface_hub import HfApi, login

'''
HuggingFace allows public repos up to 300GB, and you can contact for discrentionary additional storage:
https://huggingface.co/docs/hub/storage-limits

TODO: Even if we get the dataset on huggingface, there is still the question of how to load it in to train on

Oscar personal limit seems to be about 125GB, and I'm using 96GB for old sfno-sai stuff
Serre's lab has like >70TB uploaded
'''

client = cdsapi.Client()
# years = [str(yyyy) for yyyy in range(1979, 2026)]
years = [str(yyyy) for yyyy in range(2024, 2026)]
months = [str(mm) for mm in range(1, 13)]
request = {
    "product_type": ["reanalysis"],
    # "month": [str(mm) for mm in range(1, 13)],
    "day": [str(dd) for dd in range(1, 32)],
    "time": ["00:00", "06:00", "12:00", "18:00"],
    "data_format": "grib",
    "download_format": "unarchived"
}
cwd = os.getcwd()
load_dotenv()
hf_token = os.getenv('HF_TOKEN')
login(hf_token)
hfapi = HfApi()

def get_new_var_multiple_years(request: dict, dataset: str, vertical_levels: bool = False) -> xr.Dataset:
    global client
    global years
    global cwd
    all_data = None
    filepaths = []
    # print(request) # won't include years
    for year in years:
        request["year"] = year
        filename = client.retrieve(dataset, request).download()
        filepaths.append(f'{cwd}/{filename}')
        one_year_data = xr.open_dataset(filepaths[-1], engine='cfgrib', backend_kwargs={'indexpath':''})
        if vertical_levels:
            levels = one_year_data.isobaricInhPa.values
            var_names = [name for name in one_year_data.data_vars]
            assert len(var_names) == 1
            var_name = var_names[0]
            for level in levels:
                one_year_data[f'{var_name}{int(level)}'] = one_year_data.sel(isobaricInhPa = level)[var_name]
            one_year_data = one_year_data.drop_vars(var_name)
            one_year_data = one_year_data.drop_vars('isobaricInhPa')
        if all_data:
            all_data = xr.concat([all_data, one_year_data], "time")
        else:
            all_data = one_year_data
        del one_year_data
        os.remove(filepaths[-1])
    # for filepath in filepaths:
    #     os.remove(filepath)
    return all_data

def get_new_var(request: dict, dataset: str, vertical_levels: bool = False) -> (xr.Dataset, str):
    global client
    global cwd
    filename = client.retrieve(dataset, request).download()
    filepath = f'{cwd}/{filename}'
    data = xr.open_dataset(filepath, engine='cfgrib', backend_kwargs={'indexpath':''})
    if vertical_levels:
        levels = data.isobaricInhPa.values
        var_names = [name for name in data.data_vars]
        assert len(var_names) == 1
        var_name = var_names[0]
        if levels.shape == ():
            data[f'{var_name}{int(levels)}'] = data[var_name]
        else:
            for level in levels:
                data[f'{var_name}{int(level)}'] = data.sel(isobaricInhPa = level)[var_name]
        data = data.drop_vars(var_name)
        data = data.drop_vars('isobaricInhPa')
    return data, filepath

def get_data_by_month():
    for year in years:
        request["year"] = year
        for month in months:
            request["month"] = month
            filepaths = []

            # TODO: remove
            if year == "2024" and (month in months[:1]):
                continue

            # single pressure levels
            dataset = "reanalysis-era5-single-levels"

            request["variable"] = ["10m_u_component_of_wind"]
            u10m, filepath = get_new_var(request, dataset)
            filepaths.append(filepath)
            ds = u10m
            del u10m
            # print(f'cwd: {cwd}')
            # print(f'path of ckpt file to save: {cwd}/ckpt_u10m.zarr')
            ds.to_zarr(f'{cwd}/ckpt_u10m.zarr')

            request["variable"] = ["10m_v_component_of_wind"]
            v10m, filepath = get_new_var(request, dataset)
            filepaths.append(filepath)
            ds = xr.merge([ds, v10m])
            del v10m
            ds.to_zarr(f'{cwd}/ckpt_v10m.zarr')
            rmtree(f'{cwd}/ckpt_u10m.zarr')

            request["variable"] = ["2m_temperature"]
            t2m, filepath = get_new_var(request, dataset)
            filepaths.append(filepath)
            ds = xr.merge([ds, t2m])
            del t2m
            ds.to_zarr(f'{cwd}/ckpt_t2m.zarr')
            rmtree(f'{cwd}/ckpt_v10m.zarr')

            request["variable"] = ["mean_sea_level_pressure"]
            mslp, filepath = get_new_var(request, dataset)
            filepaths.append(filepath)
            ds = xr.merge([ds, mslp])
            del mslp
            ds.to_zarr(f'{cwd}/ckpt_mslp.zarr')
            rmtree(f'{cwd}/ckpt_t2m.zarr')

            request["variable"] = ["surface_pressure"]
            sp, filepath = get_new_var(request, dataset)
            filepaths.append(filepath)
            ds = xr.merge([ds, sp])
            del sp
            ds.to_zarr(f'{cwd}/ckpt_sp.zarr')
            rmtree(f'{cwd}/ckpt_mslp.zarr')

            request["variable"] = ["100m_u_component_of_wind"]
            u100m, filepath = get_new_var(request, dataset)
            filepaths.append(filepath)
            ds = xr.merge([ds, u100m])
            del u100m
            ds.to_zarr(f'{cwd}/ckpt_u100m.zarr')
            rmtree(f'{cwd}/ckpt_sp.zarr')

            request["variable"] = ["100m_v_component_of_wind"]
            v100m, filepath = get_new_var(request, dataset)
            filepaths.append(filepath)
            ds = xr.merge([ds, v100m])
            del v100m
            ds.to_zarr(f'{cwd}/ckpt_v100m.zarr')
            rmtree(f'{cwd}/ckpt_u100m.zarr')

            request["variable"] = ["total_column_water_vapour"]
            tcwv, filepath = get_new_var(request, dataset)
            filepaths.append(filepath)
            ds = xr.merge([ds, tcwv])
            del tcwv
            ds.to_zarr(f'{cwd}/ckpt_tcwv.zarr')
            rmtree(f'{cwd}/ckpt_v100m.zarr')

            # multiple pressure levels
            dataset = "reanalysis-era5-pressure-levels"

            # geopotential across vertical pressure levels
            request["variable"] = ["geopotential"]
            request["pressure_level"] = ["50", "250", "500", "850", "1000"]
            z_data, filepath = get_new_var(request, dataset, vertical_levels=True)
            filepaths.append(filepath)
            ds = xr.merge([ds, z_data])
            del z_data
            ds.to_zarr(f'{cwd}/ckpt_z.zarr')
            rmtree(f'{cwd}/ckpt_tcwv.zarr')

            # wind components across vertical pressure levels
            request["pressure_level"] = ["250", "500", "850", "1000"]
            request["variable"] = ["u_component_of_wind"]
            u_data, filepath = get_new_var(request, dataset, vertical_levels=True) 
            filepaths.append(filepath)
            ds = xr.merge([ds, u_data])
            del u_data
            ds.to_zarr(f'{cwd}/ckpt_u.zarr')
            rmtree(f'{cwd}/ckpt_z.zarr')
            request["variable"] = ["v_component_of_wind"]
            v_data, filepath = get_new_var(request, dataset, vertical_levels=True)
            filepaths.append(filepath)
            ds = xr.merge([ds, v_data])
            del v_data
            ds.to_zarr(f'{cwd}/ckpt_v.zarr')
            rmtree(f'{cwd}/ckpt_u.zarr')

            # temperature across vertical pressure levels
            request["pressure_level"] = ["100", "250", "500", "850"]
            request["variable"] = ["temperature"]
            t_data, filepath = get_new_var(request, dataset, vertical_levels=True)
            filepaths.append(filepath)
            ds = xr.merge([ds, t_data])
            del t_data
            ds.to_zarr(f'{cwd}/ckpt_t.zarr')
            rmtree(f'{cwd}/ckpt_v.zarr')

            # relative humidity across vertical pressure levels
            request["pressure_level"] = ["500"]
            request["variable"] = ["relative_humidity"]
            r_data, filepath = get_new_var(request, dataset, vertical_levels=True)
            filepaths.append(filepath)
            ds = xr.merge([ds, r_data])
            del r_data
            ds.to_zarr(f'{cwd}/ckpt_r.zarr')
            rmtree(f'{cwd}/ckpt_t.zarr')

            # got all data
            filename = f'era5_26vars_{year}_{month}.zarr'
            datapath = f'{cwd}/data'
            os.rename(f'{cwd}/ckpt_r.zarr', f'{datapath}/{filename}')

            # TODO: normalize data? because it's era5?

            # upload to hugging face for faster access, then delete locally            
            hfapi.upload_large_folder(
                folder_path=datapath,
                # path_in_repo=f'data/{filename}',
                repo_id="nmasi/era5",
                repo_type="dataset",
                # token = hf_token,
            )
            rmtree(f'{datapath}/{filename}')
            rmtree(f'{datapath}/.cache')

            # remove filepaths from downloaded yearxmonth data
            for filepath in filepaths:
                os.remove(filepath)

if __name__ == '__main__':
    get_data_by_month()
