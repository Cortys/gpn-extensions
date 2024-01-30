#!/usr/bin/env -S bb -m eval
(ns eval
  (:require [babashka.fs :as fs]
            [babashka.process :refer [shell]]
            [cheshire.core :as json]
            [cli-matic.core :as cli]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [taoensso.timbre :as log])
  (:import [clojure.lang ExceptionInfo]))

(def default-ds {:run.log "False"
                 :run.num_inits 10
                 :run.num_splits 10})
(def big-default-ds (merge default-ds
                           {:model.sparse_propagation "True"}))
(def datasets
  {"CoraML" default-ds
   "CiteSeerFull" default-ds
   "AmazonPhotos" big-default-ds
   "AmazonComputers" big-default-ds
   "PubMedFull" big-default-ds
   "ogbn-arxiv" (merge big-default-ds
                       {:data.split "public"
                        :run.num_splits 1
                        :run.reduced_training_metrics "True"
                        :training.eval_every 10
                        :training.stopping_patience 5
                        :run.log "True"})})

(def models
  {"appnp" {::name "appnp"}
   "ggp" {::name "ggp"}
   "gdk" {::name "gdk"}
   "gpn" {::name "gpn_16"}
   "gpn_rw" {::name "gpn_16"
             :model.adj_normalization "rw"}
   "gpn_lop" {::name "gpn_16"
              :model.model_name "GPN_LOP"
              :model.sparse_x_prune_threshold 0.01}})

(def settings
  {"classification" {}
   "ood_loc" {}
   "ood_features_normal" {::name "ood_features"
                          ::depends-on "classification"
                          :data.ood_perturbation_type "normal"}
   "ood_features_ber" {::name "ood_features"
                       ::depends-on "classification"
                       :data.ood_perturbation_type "bernoulli_0.5"}})

(def combination-overrides
  {{::model "gpn_lop" ::dataset "PubMedFull"}
   {:run.log "True"}
   {::model "gpn_lop" ::dataset "ogbn-arxiv"}
   {:model.sparse_x_prune_threshold 0.01}})

(def default-datasets ["CoraML"
                       "CiteSeerFull"
                       "AmazonPhotos"
                       "AmazonComputers"
                       "PubMedFull"
                       #_"ogbn-arxiv"])
(def default-models ["appnp"
                     #_"ggp"
                     #_"gdk"
                     "gpn"
                     "gpn_rw"
                     "gpn_lop"])
(def default-settings ["classification"
                       "ood_loc"])

(defn run-config!
  [config & {:keys [::depends-on] :as overrides}]
  (let [args (keep (fn [[k v]]
                     (when-not (namespace k)
                       (str (name k) "=" v)))
                   overrides)]
    (when depends-on
      (log/info "Running dependency...")
      (apply run-config! depends-on)
      (log/info "Ran dependency. Continuing with parent..."))
    (log/info "Running with" config (str/join " " args))
    (try
      (apply shell "python3 train_and_eval.py" "--force"
             "with" config args)
      (catch ExceptionInfo _
        (log/error "Run failed:" config (str/join " " args)))))
  )

(defn build-config-path
  [_ model setting]
  (let [model-name (::name model)
        setting-name (::name setting)
        dir (cond
              (str/includes? model-name "gpn") "gpn"
              :else "reference")
        path (str "configs/" dir "/" setting-name "_" model-name ".yaml")]
    (assert (fs/exists? path))
    path))

(defn build-config-cli-params
  [dataset-name model-name setting-name overrides]
  (assert (contains? datasets dataset-name))
  (assert (contains? models model-name))
  (assert (contains? settings setting-name))
  (let [dataset (merge {::name dataset-name
                        :data.dataset dataset-name}
                       (datasets dataset-name))
        model (merge {::name model-name} (models model-name))
        setting (merge {::name setting-name} (settings setting-name))
        setting-dep (::depends-on setting)
        combination-overrides
        (apply merge
               (for [d [nil [::dataset dataset-name]]
                     m [nil [::model model-name]]
                     s [nil [::setting setting-name]]
                     :let [c (combination-overrides (into {} (remove nil?) [d m s]))]
                     :when c]
                 c))]
    [(build-config-path dataset model setting)
     ::depends-on (when setting-dep
                    (build-config-cli-params dataset-name
                                             model-name
                                             setting-dep
                                             overrides))
     (dissoc (merge dataset model setting combination-overrides overrides)
             ::name)]))

(defn stringify-combination
  [dataset-name model-name setting-name]
  (str setting-name "/" dataset-name "/" model-name))

(defn run-combination!
  [dataset-name model-name setting-name overrides]
  (apply run-config! (build-config-cli-params dataset-name
                                              model-name
                                              setting-name
                                              overrides)))

(defn cached-run-combination!
  [dataset-name model-name setting-name overrides
   & {:keys [only-cached] :or {only-cached false}}]
  (let [combination-id (stringify-combination dataset-name
                                              model-name
                                              setting-name)
        results-path (str "results/" combination-id ".json")]
    (if (and (not (:run.reeval overrides)) (fs/exists? results-path))
      (do
        (log/info "Loading" combination-id "from cache...")
        (json/parse-stream (io/reader results-path)))
      (if only-cached
        (throw (Exception. (str "No cached results for" combination-id)))
        (do
          (log/info "Running" combination-id "...")
          (fs/create-dirs (fs/parent results-path))
          (run-combination! dataset-name
                            model-name
                            setting-name
                            (assoc overrides :run.results_path results-path)))))))

(defn run-combinations!
  [dataset-names model-names setting-names overrides & {:as opts}]
  (doseq [dataset-name dataset-names
          model-name model-names
          setting-name setting-names]
    (cached-run-combination! dataset-name model-name setting-name overrides opts)))

(defn print-grid
  [dataset-names model-names setting-names overrides]
  (println "Datasets:")
  (doseq [dataset dataset-names]
    (println "-" dataset))
  (println "\nModels:")
  (doseq [model model-names]
    (println "-" model))
  (println "\nSettings:")
  (doseq [setting setting-names]
    (println "-" setting))
  (println "\nOverrides:")
  (doseq [[k v] overrides]
    (println "-" k "=" v)))

(defn parse-override
  [override]
  (let [[k v] (str/split override #"=" 2)
        k (str/trim k)
        v (str/trim v)]
    (assert k)
    (assert v)
    [(keyword k) v]))

(defn run-eval!
  [{:keys [dataset model setting override
           dry retrain reeval only-cached]
    :or {dataset default-datasets
         model default-models
         setting default-settings}}]
  (let [default-config (cond-> {}
                         retrain (assoc :run.retrain true)
                         reeval (assoc :run.reeval true))
        override (into default-config (map parse-override) override)]
    (print-grid dataset model setting override)
    (when-not dry
      (println "\nStarting experiments...\n")
      (Thread/sleep 500)
      (run-combinations! dataset model setting override
                         :only-cached only-cached))
    (println "\nDone.")))

(defn run-table-gen!
  []
  )

(def CLI-CONFIGURATION
  {:command "cuq-gnn"
   :description "An evaluation script."
   :version "0.1.0"
   :subcommands [{:command "eval"
                  :description "Run experiments."
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
                         {:as "Overrides"
                          :option "override"
                          :short "o"
                          :type :string
                          :multiple true}
                         {:as "Dry Run"
                          :option "dry"
                          :default false
                          :type :with-flag}
                         {:as "Retrain Models"
                          :option "retrain"
                          :default false
                          :type :with-flag}
                         {:as "Reevaluate Models"
                          :option "reeval"
                          :default false
                          :type :with-flag}
                         {:as "Only Cached"
                          :option "only-cached"
                          :default false
                          :type :with-flag}]
                  :runs run-eval!}
                 {:command "table"
                  :description "Generate a results CSV."
                  :runs run-table-gen!}]})

(defn -main
  [& args]
  (cli/run-cmd (rest args) CLI-CONFIGURATION))

(comment
  (run-config! "configs/gpn/classification_gpn_16.yaml"
               :data.dataset "CoraML")
  (cli/run-cmd* [] CLI-CONFIGURATION)
  )
