from nuscenes import NuScenes


class NuScenesTables(NuScenes):
    """Load LiDAR and annotation tables for conversion and evaluation."""

    def __init__(self, version, dataroot, verbose=True):
        self.version = version
        self.dataroot = dataroot
        self.verbose = verbose
        self.table_names = [
            "category", "attribute", "visibility", "instance", "sensor",
            "calibrated_sensor", "ego_pose", "log", "scene", "sample",
            "sample_data", "sample_annotation",
        ]
        self._token2ind = {}
        for name in self.table_names:
            rows = self.__load_table__(name)
            setattr(self, name, rows)
            self._token2ind[name] = {row["token"]: index for index, row in enumerate(rows)}
            if verbose:
                print(f"{len(rows)} {name}")

        # Keep the annotation and sensor indexes from nuScenes devkit 1.1.10.
        for annotation in self.sample_annotation:
            instance = self.get("instance", annotation["instance_token"])
            annotation["category_name"] = self.get("category", instance["category_token"])["name"]
        for sample in self.sample:
            sample["data"] = {}
            sample["anns"] = []
        for data in self.sample_data:
            calibration = self.get("calibrated_sensor", data["calibrated_sensor_token"])
            sensor = self.get("sensor", calibration["sensor_token"])
            data["sensor_modality"] = sensor["modality"]
            data["channel"] = sensor["channel"]
            if data["is_key_frame"]:
                sample = self.get("sample", data["sample_token"])
                sample["data"][data["channel"]] = data["token"]
        for annotation in self.sample_annotation:
            self.get("sample", annotation["sample_token"])["anns"].append(annotation["token"])
