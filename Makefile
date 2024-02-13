# use "make all" to build everything for all datasets
# use "make all_<dataset>" to only build a single dataset
# use "make <target>" to build a specific target while ignoring the dependent targets

.PHONY: all all_toy_data  all_mnist all_nfbs template_toy_data resolution_measure_toy_data visualization_toy_data template_mnist resolution_measure_mnist visualization_mnist normalize_nfbs template_nfbs resolution_measure_nfbs visualization_nfbs requirements clean_data clean_plots


python		:= intelpython3 # set custom python interpreter
data_raw	:= data/raw

# Matplotlib backend: Agg (no gui)
export MPLBACKEND := Agg

# use gpu for computation of template resolution measure: 1 (yes) or 0 (no)
export TR_USE_GPU := 1


all: all_toy_data all_mnist all_nfbs

all_toy_data: template_toy_data resolution_measure_toy_data visualization_toy_data

all_mnist: template_mnist resolution_measure_mnist visualization_mnist

all_nfbs: normalize_nfbs template_nfbs resolution_measure_nfbs visualization_nfbs


$(data_raw)/NFBS_Dataset:
	$(python) -m src.data.download_NFBS

$(data_raw)/MNIST:
	$(python) -m src.data.download_MNIST

$(data_raw)/toy_data_1d:
	$(python) -m src.data.create_toy_data_1d

normalize_nfbs: $(data_raw)/NFBS_Dataset
	$(python) -m src.data.normalize_NFBS


template_nfbs:
	cd src; $(python) -m templates.templates_NFBS

template_mnist: $(data_raw)/MNIST
	cd src; $(python) -m templates.templates_MNIST

template_toy_data: $(data_raw)/toy_data_1d
	cd src; $(python) -m templates.templates_toy_data


resolution_measure_nfbs:
	cd src; $(python) -m analysis.template_resolution_NFBS

resolution_measure_mnist:
	cd src; $(python) -m analysis.template_resolution_MNIST

resolution_measure_toy_data:
	cd src; $(python) -m analysis.template_resolution_toy_data


visualization_nfbs:
	cd src; $(python) -m visualization.visualize_NFBS

visualization_mnist:
	cd src; $(python) -m visualization.visualize_MNIST

visualization_toy_data:
	cd src; $(python) -m visualization.visualize_toy_data


requirements:
	$(python) -m pip install -r requirements.txt


clean_data:
	rm -rf data

clean_plots:
	rm -rf plots
