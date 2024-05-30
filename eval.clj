#!/usr/bin/env -S bb -m eval
(ns eval
  (:require [babashka.fs :as fs]
            [babashka.process :refer [shell]]
            [cheshire.core :as json]
            [cli-matic.core :as cli]
            [clojure.java.io :as io]
            [clojure.math :as math]
            [clojure.string :as str]
            [taoensso.timbre :as log])
  (:import [clojure.lang ExceptionInfo]))

;; Config

(def default-ds {:run.log "False"
                 :run.num_inits 10
                 :run.num_splits 10})
(def big-default-ds (merge default-ds
                           {:model.sparse_propagation "True"}))
(def datasets
  {"CoraML" default-ds
   "CiteSeerFull" (assoc default-ds ::colname "CiteSeer" ::inline-name "CiteSeer")
   "AmazonPhotos" (assoc big-default-ds ::colname "Photos" ::inline-name "Amazon\\\\Photos")
   "AmazonComputers" (assoc big-default-ds ::colname "Computers" ::inline-name "Amazon\\\\Computers")
   "PubMedFull" (assoc big-default-ds ::colname "PubMed" ::inline-name "PubMed")
   "ogbn-arxiv" (merge big-default-ds
                       {::colname "Arxiv"
                        ::inline-name "OGBN\\\\Arxiv"
                        :data.split "public"
                        :run.num_splits 1
                        :run.reduced_training_metrics "True"
                        :training.eval_every 10
                        :training.stopping_patience 5
                        :model.entropy_num_samples 100
                        :run.log "True"})})

(def models
  {"appnp" {::name "appnp"
            ::inline-name "APPNP"
            ::ignored-metrics [:ood_detection_aleatoric_entropy_auroc]}
   "ggp" {::name "ggp"
          ::inline-name "GGP"
          :run.num_inits 1}
   "matern_ggp" {::name "matern_ggp"
                 ::colname "maternGGP"
                 ::inline-name "Matern-GGP"
                 :run.num_inits 1}
   "gdk" {::name "gdk" ::inline-name "GKDE"}
   "gpn" {::name "gpn_16" ::inline-name "GPN (sym)"}
   "gpn_rw" {::name "gpn_16"
             ::colname "gpnRW"
             ::inline-name "GPN (rw)"
             :model.adj_normalization "rw"}
   "gpn_lop" {::name "gpn_16"
              ::inline-name "LOP-GPN"
              ::colname "gpnLOP"
              :model.model_name "GPN_LOP"
              :model.sparse_x_prune_threshold 0.01}})

(def settings
  {"classification" {}
   "ood_loc" {::colname "oodLoc"}
   "ood_features_normal" {::name "ood_features"
                          ::colname "oodNormal"
                          ::depends-on "classification"
                          :data.ood_perturbation_type "normal"
                          :run.experiment_name "ood_features_normal"}
   "ood_features_ber" {::name "ood_features"
                       ::colname "oodBer"
                       ::depends-on "classification"
                       :data.ood_perturbation_type "bernoulli_0.5"
                       :run.experiment_name "ood_features_ber"}})

(def combination-overrides
  {{::model "gpn_lop" ::dataset "PubMedFull"}
   {:run.log "True"}
   {::model "gpn_lop" ::dataset "ogbn-arxiv"}
   {:model.sparse_x_prune_threshold 0.01
    :run.num_inits 2}
   {::model "gdk" ::dataset "ogbn-arxiv"}
   {:model.gdk_cutoff 2}
   {::model "matern_ggp" ::dataset "ogbn-arxiv"}
   {::skip true}})

(def default-datasets ["CoraML"
                       "CiteSeerFull"
                       "AmazonPhotos"
                       "AmazonComputers"
                       "PubMedFull"
                       "ogbn-arxiv"])
(def default-models ["appnp"
                     "matern_ggp"
                     "gdk"
                     "gpn"
                     "gpn_rw"
                     "gpn_lop"])
(def default-settings ["classification"
                       "ood_loc"
                       "ood_features_normal"
                       "ood_features_ber"])

;; Utils

(defn run-config!
  [config & {:as overrides}]
  (let [args (keep (fn [[k v]]
                     (when-not (namespace k)
                       (str (name k) "=" v)))
                   overrides)]
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
        combination-overrides
        (apply merge
               (for [d [nil [::dataset dataset-name]]
                     m [nil [::model model-name]]
                     s [nil [::setting setting-name]]
                     :let [c (combination-overrides (into {} (remove nil?) [d m s]))]
                     :when c]
                 c))]
    [(build-config-path dataset model setting)
     (dissoc (merge dataset model setting combination-overrides overrides)
             ::name)]))

(defn stringify-combination
  [dataset-name model-name setting-name]
  (str setting-name "/" dataset-name "/" model-name))

(defn- valid-var?
  [var]
  (cond
    (vector? var) (every? valid-var? var)
    (number? var) (not (Double/isNaN var))
    :else false))

(defn recurse
  [f x]
  (if (vector? x)
    (mapv (partial recurse f) x)
    (f x)))

(defn add-ses-to-var-map
  [samples var-map]
  (into var-map
        (comp (filter #(-> % first name (str/ends-with? "_var")))
              (filter #(-> % second valid-var?))
              (keep (fn [[k v]]
                      (let [k (name k)
                            k (keyword (str (subs k 0 (- (count k) 4)) "_se"))]
                        (when-not (contains? var-map k)
                          [k (recurse #(math/sqrt (/ % samples)) v)])))))
        var-map))

(defn update-cached-results
  [results overrides]
  (let [samples (* (:run.num_splits overrides) (:run.num_inits overrides))
        results (update-vals results (partial add-ses-to-var-map samples))]
    results))

(defn get-results
  [dataset-name model-name setting-name overrides
   & {:keys [only-cached no-cache delete]
      :or {only-cached false no-cache false delete false}}]
  (assert (not (and only-cached no-cache))
          "only-cached and no-cache cannot be enabled at the same time.")
  (assert (not (and only-cached delete))
          "only-cached and delete cannot be enabled at the same time.")
  (let [combination-id (stringify-combination dataset-name
                                              model-name
                                              setting-name)
        results-path (str "results/" combination-id ".json")
        [_ config :as params]
        (build-config-cli-params dataset-name
                                 model-name
                                 setting-name
                                 (assoc overrides :run.results_path results-path))]
    (when (::skip config)
      (throw (ex-info (str "Skipped " combination-id ".")
                      {::cause :skip
                       :dataset-name dataset-name
                       :model-name model-name
                       :setting-name setting-name
                       :overrides overrides})))
    (if (and (not (:run.retrain overrides))
             (not (:run.reeval overrides))
             (not no-cache) (not delete)
             (fs/exists? results-path))
      (log/debug "Loading" combination-id "from cache...")
      (if only-cached
        (throw (ex-info (str "No cached results for " combination-id)
                        {::cause :no-cache
                         :dataset-name dataset-name
                         :model-name model-name
                         :setting-name setting-name
                         :overrides overrides}))
        (do
          (log/info (if delete "Deleting" "Running")
                    combination-id "...")
          (fs/create-dirs (fs/parent results-path))
          (apply run-config! params))))
    (when-not delete
      (let [results (json/parse-stream (io/reader results-path) true)
            results (update-cached-results results config)
            results (update-vals results #(apply dissoc % (::ignored-metrics config)))]
        results))))

(defn try-get-results
  [& args]
  (try
    (apply get-results args)
    (catch clojure.lang.ExceptionInfo e
      (if (= (::cause (ex-data e)) :skip)
        (log/debug (ex-message e))
        (log/error (ex-message e)))
      nil)))

;; Run experiments

(defn run-combinations!
  [dataset-names model-names setting-names overrides & {:as opts}]
  (doseq [dataset-name dataset-names
          model-name model-names
          setting-name setting-names]
    (try-get-results dataset-name model-name setting-name overrides opts)))

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
           dry retrain reeval only-cached cache delete]
    :or {dataset default-datasets
         model default-models
         setting default-settings}}]
  (let [default-config (cond-> {}
                         retrain (assoc :run.retrain true)
                         reeval (assoc :run.reeval true)
                         delete (assoc :run.delete_run true))
        override (into default-config (map parse-override) override)]
    (print-grid dataset model setting override)
    (when delete
      (print "\nAre you sure you want to delete all results listed above? (y/N) ")
      (flush)
      (let [input (read-line)]
        (if (str/starts-with? (str/lower-case input) "y")
          (do
            (log/info "Will start deleting all selected results in 3s...")
            (Thread/sleep 1000)
            (log/info "Will start deleting all selected results in 2s...")
            (Thread/sleep 1000)
            (log/info "Will start deleting all selected results in 1s...")
            (Thread/sleep 1000)
            (log/info "Deleting..."))
          (do
            (log/error "Aborted.")
            (System/exit 1)))))
    (when-not dry
      (log/info (str (if delete "Deleting" "Starting")
                     " experiments ("
                     "only-cached=" only-cached ", "
                     "cache=" cache
                     ")..."))
      (Thread/sleep 500)
      (run-combinations! dataset model setting override
                         :only-cached only-cached
                         :no-cache (not cache)
                         :delete delete))
    (log/info "Done.")))

;; Accuracy-rejection tables

(defn get-acc-rej-curve
  [dataset-name model-name confidence-type uncertainty-type]
  (let [key (str "accuracy_rejection_" confidence-type "_confidence_"
                 uncertainty-type)
        mean-kw (keyword key)
        var-kw (keyword (str key "_var"))
        se-kw (keyword (str key "_se"))
        results (try-get-results dataset-name model-name
                                 "classification" {}
                                 :only-cached true)
        results (:test results)
        mean (mean-kw results)
        var (var-kw results)
        se (se-kw results)]
    (when mean
      {:mean mean, :var var, :se se})))

(defn get-acc-rej-curve-with-fallback
  [dataset-name model-name types]
  (first (eduction (keep #(apply get-acc-rej-curve dataset-name model-name %))
                   types)))

(defn run-acc-rej-table-gen!
  [dataset type]
  (let [types (case type
                "sample"
                [["sample" "epistemic"]
                 ["sample" "aleatoric"]]
                "prediction"
                [["prediction" "total"]
                 ["prediction" "epistemic"]
                 ["prediction" "aleatoric"]]
                "sample_total"
                [["sample" "total"]]
                "sample_total_entropy"
                [["sample" "total_entropy"]]
                "sample_aleatoric"
                [["sample" "aleatoric"]]
                "sample_aleatoric_entropy"
                [["sample" "aleatoric_entropy"]]
                "sample_epistemic"
                [["sample" "epistemic"]]
                "sample_epistemic_entropy"
                [["sample" "epistemic_entropy"]]
                "sample_epistemic_entropy_diff"
                [["sample" "epistemic_entropy_diff"]]
                "prediction_aleatoric"
                [["prediction" "aleatoric"]]
                "prediction_epistemic"
                [["prediction" "epistemic"]])
        curves
        (into {}
              (comp (map #(do [% (get-acc-rej-curve-with-fallback dataset %
                                                                  types)]))
                    (filter (comp :mean second)))
              default-models)
        model-names (keys curves)
        N (-> curves first second :mean count)
        head (str/join "," (into ["p"]
                                 (comp
                                  (map #(-> % models (::colname %)))
                                  (mapcat #(do [(str % "Mean")
                                                (str % "Var")
                                                (str % "SE")])))
                                 model-names))
        body (for [i (range N)
                   :let [p (double (/ i (dec N)))]]
               (str/join ","
                         (into [p]
                               (mapcat (fn [model]
                                         (let [{:keys [mean var se]}
                                               (curves model)]
                                           [(get mean i)
                                            (get var i 0)
                                            (get se i 0)]))
                                       model-names))))
        csv (str/join "\n" (cons head body))]
    (spit (str "tables/acc_rej_" type "_" (-> dataset datasets (::colname dataset)) ".csv") csv)))

(defn run-acc-rej-tables-gen!
  [& _]
  (log/info "Generating accuracy-rejection tables...")
  (doseq [dataset default-datasets
          type ["sample" "prediction"
                "sample_total" "sample_total_entropy"
                "sample_aleatoric" "sample_aleatoric_entropy"
                "sample_epistemic" "sample_epistemic_entropy"
                "sample_epistemic_entropy_diff"
                "prediction_aleatoric" "prediction_epistemic"]]
    (run-acc-rej-table-gen! dataset type))
  (log/info "Done."))

;; ID-OOD table

(defn compute-certainty-change
  ([results uncertainty-type]
   (compute-certainty-change results uncertainty-type false))
  ([results uncertainty-type total-norm]
   (let [id-certainty (keyword (str "id_avg_sample_confidence_" uncertainty-type))
         ood-certainty (keyword (str "ood_avg_sample_confidence_" uncertainty-type))

         id-certainty (id-certainty results)
         ood-certainty (ood-certainty results)
         norm (if total-norm
                (:id_avg_sample_confidence_total_entropy results)
                id-certainty)]
     #_(println uncertainty-type ood-certainty id-certainty norm)
     (when (and ood-certainty norm)
       (if (not= (math/signum ood-certainty) (math/signum norm))
         (do (log/warn "Sign mismatch" {:uncertainty-type uncertainty-type
                                        :ood-certainty ood-certainty
                                        :norm norm
                                        :total-norm total-norm})
             nil)
         (dec (/ (- ood-certainty) (Math/abs norm))))))))

(defn round
  ([n places & {:keys [factor decimals sign]
                :or {factor 1
                     decimals (dec places)
                     sign false}}]
   (when n
     (let [scale (Math/pow 10 decimals)
           num (/ (Math/round (* n scale factor)) scale)
           res (format (str "%." decimals "f") num)
           l (inc places)
           l (if (and (not sign) (pos? num)) l (inc l))
           l (min (count res) l)
           res (subs res 0 l)]
       (if (and sign (pos? num))
         (str "+" res)
         res)))))

(defn get-se
  [results metric]
  (let [se (keyword (str metric "_se"))]
    (se results 0)))

(defn get-metric
  [results best-results metric]
  (let [result (metric results)
        best-result (best-results metric)]
    [(round result 4 :decimals 2 :factor 100)
     (round (get-se results metric) 4 :decimals 2 :factor 100)
     (if (and result best-result (>= result best-result)) "1" "0")]))

(defn run-id-ood-table-gen!
  [& _]
  (log/info "Generating ID-OOD table...")
  (let [dataset-names default-datasets
        model-names default-models
        setting-names (rest default-settings)
        class-metrics [:accuracy]
        settings-metrics [:id_accuracy
                          :ood_accuracy
                          :ood_detection_total_auroc
                          :ood_detection_total_entropy_auroc
                          :ood_detection_aleatoric_auroc
                          :ood_detection_aleatoric_entropy_auroc
                          :ood_detection_epistemic_auroc
                          :ood_detection_epistemic_entropy_auroc
                          :ood_detection_epistemic_entropy_diff_auroc]
        cols (into ["id" "dataset" "model" "acc" "accSE" "accBest"]
                   (mapcat (fn [setting]
                             (let [setting (-> setting settings ::colname)]
                               (concat
                                (mapcat #(do [% (str % "SE") (str % "Best")])
                                        [(str setting "IdAcc")
                                         (str setting "OodAcc")
                                         (str setting "OodTotal")
                                         (str setting "OodTotalEntropy")
                                         (str setting "OodAleatoric")
                                         (str setting "OodAleatoricEntropy")
                                         (str setting "OodEpistemic")
                                         (str setting "OodEpistemicEntropy")
                                         (str setting "OodEpistemicEntropyDiff")])
                                [(str setting "TotalEntropyChange")
                                 (str setting "AleatoricEntropyChange")
                                 (str setting "EpistemicEntropyChange")
                                 (str setting "EpistemicEntropyDiffChange")]))))
                   setting-names)
        head (str/join "," cols)
        rows (for [dataset dataset-names
                   model model-names
                   :let [class-results
                         (-> (try-get-results dataset model "classification" {}
                                              :only-cached true)
                             :test)
                         setting-results (zipmap setting-names
                                                 (map (fn [setting]
                                                        (-> (try-get-results dataset model setting {}
                                                                             :only-cached true)
                                                            :test))
                                                      setting-names))]]
               {:dataset dataset
                :model model
                :class-results class-results
                :setting-results setting-results})
        metrics (concat (map #(do [:class-results %]) class-metrics)
                        (for [setting setting-names
                              metric settings-metrics]
                          [:setting-results setting metric]))
        row-groups (group-by :dataset rows)
        best-metric-groups
        (into {}
              (mapcat (fn [[group rows]]
                        (map (fn [metric]
                               (let [vals (keep #(get-in % metric) rows)
                                     best-val (apply max ##-Inf vals)]
                                 [(conj metric group) best-val]))
                             metrics)))
              row-groups)
        body (map-indexed
              (fn [i {:keys [dataset model class-results setting-results]}]
                (let [row [i
                           (-> dataset datasets (::inline-name dataset))
                           (-> model models (::inline-name model))]
                      class-best-results (fn [metric] (best-metric-groups [:class-results metric dataset]))
                      row (into row
                                (mapcat #(get-metric class-results class-best-results %))
                                class-metrics)]
                  (str/join ","
                            (into row
                                  (mapcat (fn [[setting results]]
                                            (let [best-results (fn [metric]
                                                                 (best-metric-groups [:setting-results setting
                                                                                      metric dataset]))
                                                  total-entropy-change (compute-certainty-change results "total_entropy")
                                                  aleatoric-entropy-change (compute-certainty-change results "aleatoric_entropy")
                                                  epistemic-entropy-change (compute-certainty-change results "epistemic_entropy")
                                                  epistemic-entropy-diff-change (compute-certainty-change results "epistemic_entropy_diff")]
                                              (concat (mapcat #(get-metric results best-results %)
                                                              settings-metrics)
                                                      (map #(round % 4 :factor 100 :sign true)
                                                           [total-entropy-change
                                                            aleatoric-entropy-change
                                                            epistemic-entropy-change
                                                            epistemic-entropy-diff-change])))))
                                  setting-results))))
              rows)
        csv (str/join "\n" (cons head body))]
    (log/info (str "Creating table with " (count cols) " columns..."))
    (spit "tables/id_ood.csv" csv))
  (log/info "Done."))

;; CLI

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
                          :type :with-flag}
                         {:as "No Cache"
                          :option "cache"
                          :default true
                          :type :with-flag}
                         {:as "Delete existing models and results"
                          :option "delete"
                          :default false
                          :type :with-flag}]
                  :runs run-eval!}
                 {:command "acc-rej-tables"
                  :description "Generate acc-rej CSVs."
                  :runs run-acc-rej-tables-gen!}
                 {:command "id-ood-table"
                  :description "Generate ID-OOD CSV."
                  :runs run-id-ood-table-gen!}]})

(defn -main
  [& args]
  (cli/run-cmd (rest args) CLI-CONFIGURATION))

(comment
  (run-config! "configs/gpn/classification_gpn_16.yaml"
               :data.dataset "CoraML")
  (cli/run-cmd* [] CLI-CONFIGURATION)

  (run-acc-rej-tables-gen!)
  )
