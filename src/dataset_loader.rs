use std::fs;

#[cfg(feature = "csv-support")]
use csv::ReaderBuilder;

pub struct Dataset {
    pub pretraining_data: Vec<String>,
    pub chat_training_data: Vec<String>,
}

#[allow(dead_code)]
#[allow(clippy::upper_case_acronyms)]
pub enum DatasetType {
    JSON,
    CSV,
}

impl Dataset {
    pub fn new(
        pretraining_data_path: String,
        chat_training_data_path: String,
        type_of_data: DatasetType,
    ) -> Self {
        let pretraining_data = match type_of_data {
            DatasetType::CSV => get_data_from_csv(pretraining_data_path.clone()),
            DatasetType::JSON => get_data_from_json(pretraining_data_path.clone()),
        };

        let chat_training_data = match type_of_data {
            DatasetType::CSV => get_data_from_csv(chat_training_data_path),
            DatasetType::JSON => get_data_from_json(chat_training_data_path),
        };

        Dataset {
            pretraining_data,
            chat_training_data,
        }
    }
}

fn get_data_from_json(path: String) -> Vec<String> {
    // convert json file to Vec<String>
    match fs::read_to_string(&path) {
        Ok(data_json) => match serde_json::from_str::<Vec<String>>(&data_json) {
            Ok(data) => data,
            Err(e) => {
                log::error!("解析JSON数据失败 ({}): {}", path, e);
                Vec::new()
            }
        },
        Err(e) => {
            log::error!("读取数据文件失败 ({}): {}", path, e);
            Vec::new()
        }
    }
}

#[cfg(feature = "csv-support")]
fn get_data_from_csv(path: String) -> Vec<String> {
    // convert csv file to Vec<String>
    let file = match fs::File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            log::error!("打开CSV文件失败 ({}): {}", path, e);
            return Vec::new();
        }
    };
    let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);
    let mut data = Vec::new();

    for result in rdr.records() {
        match result {
            Ok(record) => {
                // Each record is a row, join all columns into a single string
                data.push(record.iter().collect::<Vec<_>>().join(","));
            }
            Err(e) => {
                log::warn!("读取CSV记录失败 ({}): {}", path, e);
            }
        }
    }
    data
}

#[cfg(not(feature = "csv-support"))]
fn get_data_from_csv(path: String) -> Vec<String> {
    log::warn!(
        "CSV support is not enabled. Please enable the 'csv-support' feature to use CSV files: {}",
        path
    );
    Vec::new()
}
