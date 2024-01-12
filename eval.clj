#!/usr/bin/env -S bb -m eval
(ns eval
  (:require [babashka.fs :as fs]
            [babashka.process :refer [shell]]
            [cli-matic.core :as cli]
            [clojure.string :as str]
            [taoensso.timbre :as log]) 
  (:import [clojure.lang ExceptionInfo]))

(def default-splits {:run.num_inits 10
                     :run.num_splits 1})
(def datasets
  {"CoraML" default-splits
   "AmazonPhotos" default-splits
   "ogbn-arxiv" {:run.num_inits 10}})

(def models
  {"gpn" {::name "gpn_16"}
   "gpn_lop" {::name "gpn_16"
              :model.model_name "GPN_LOP"}})

(def settings
  {"classification" {}
   "ood_loc" {}
   "ood_features_normal" {::name "ood_features"
                          ::depends-on "classification"
                          :data.ood_perturbation_type "normal"}
   "ood_features_ber" {::name "ood_features"
                       ::depends-on "classification"
                       :data.ood_perturbation_type "bernoulli_0.5"}})

(def default-datasets (keys datasets))
(def default-models (keys models))
(def default-settings ["ood_loc"])

(defn run-config!
  [config & {:keys [::depends-on] :as overrides}]
  (let [args (keep (fn [[k v]]
                     (when-not (namespace k) 
                       (str (name k) "=" v))) 
                   overrides)]
    (log/info "Running with" config (str/join " " args))
    (try 
      (apply shell "python3 train_and_eval.py"
             "with" config args)
      (catch ExceptionInfo _
        (log/error "Run failed:" config (str/join " " args) ))))
  )

(defn build-config-path
  [_ model setting]
  (let [model-name (::name model)
        setting-name (::name setting)
        dir (if (str/includes? model-name "gpn") "gpn" "reference")
        path (str "configs/" dir "/" setting-name "_" model-name ".yaml")]
    (assert (fs/exists? path))
    path))

(defn build-config-cli-params
  [dataset-name model-name setting-name]
  (assert (contains? datasets dataset-name))
  (assert (contains? models model-name))
  (assert (contains? settings setting-name))
  (let [dataset (merge {::name dataset-name} (datasets dataset-name))
        model (merge {::name model-name} (models model-name))
        setting (merge {::name setting-name} (settings setting-name))
        setting-dep (::depends-on setting)]
    [(build-config-path dataset model setting)
     ::depends-on (when setting-dep
                    (build-config-cli-params dataset-name 
                                             model-name 
                                             setting-dep))
     (dissoc (merge dataset model setting) ::name)]))

(defn run-combination!
  [dataset-name model-name setting-name]
  (apply run-config! (build-config-cli-params dataset-name
                                              model-name
                                              setting-name)))

(defn run-combinations!
  [dataset-names model-names setting-names]
  (doseq [dataset-name dataset-names
          model-name model-names
          setting-name setting-names]
    (run-combination! dataset-name model-name setting-name)))

(defn print-grid
  [dataset-names model-names setting-names]
  (println "Datasets:")
  (doseq [dataset dataset-names]
    (println "-" dataset))
  (println "\nModels:")
  (doseq [model model-names]
    (println "-" model))
  (println "\nSettings:")
  (doseq [setting setting-names]
    (println "-" setting)))

(defn start!
  [{:keys [dataset model setting dry]
    :or {dataset default-datasets
         model default-models
         setting default-settings}}]
  (print-grid dataset model setting) 
  (when-not dry
    (println "\nStarting experiments...\n")
    (Thread/sleep 500)
    (run-combinations! dataset model setting))
  (println "\nDone."))

(def CLI-CONFIGURATION
  {:command "cuq-gnn"
   :description "An evaluation script."
   :version "0.1.0"
   :opts [{:as "Datasets"
           :option "dataset"
           :short "d"
           :type :string
           :multiple true}
          {:as "Models"
           :option "model"
           :short "m"
           :type :string
           :multiple true}
          {:as "Settings"
           :option "setting"
           :short "s"
           :type :string
           :multiple true}
          {:as "Dry Run"
           :option "dry"
           :default false
           :type :with-flag}] 
   :runs start!})

(defn -main
  [& args]
  (cli/run-cmd (rest args) CLI-CONFIGURATION))

(comment
  (run-config! "configs/gpn/classification_gpn_16.yaml"
               :data.dataset "CoraML")
  (cli/run-cmd* [] CLI-CONFIGURATION)
  )
