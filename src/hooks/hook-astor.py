from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# This collects all dynamically imported scrapy modules and data files.
hiddenimports = (collect_submodules('astor') +
                 collect_submodules('astor.pipelines') +
                 collect_submodules('astor.extensions') +
                 collect_submodules('astor.utils')
)
datas = collect_data_files('astor')