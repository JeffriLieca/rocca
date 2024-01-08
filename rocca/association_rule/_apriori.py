import itertools
import pandas as pd
from collections import defaultdict

class Apriori:
    """
    Implementasi algoritma Apriori untuk analisis aturan asosiasi dalam data transaksional.
    Algoritma ini digunakan untuk menemukan frequent itemsets dan aturan asosiasi yang 
    memenuhi batas minimum support dan confidence yang ditentukan.

    Attributes:
        min_support (float): Nilai minimum support untuk menemukan frequent itemsets.
        min_confidence (float): Nilai minimum confidence untuk aturan asosiasi.
        min_lift (float): Nilai minimum lift untuk aturan asosiasi.
        verbose (bool): Jika True, menampilkan proses langkah demi langkah.
        frequent_itemsets (list): Daftar itemsets yang sering muncul dalam data.
        rules (list): Daftar aturan asosiasi yang dihasilkan.

    Metode utama:
        fit(data): Menjalankan algoritma Apriori pada dataset yang diberikan.
    """
    def __init__(self, min_support, min_confidence, min_lift, verbose=False):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.frequent_itemsets = []
        self.rules = []
        self.verbose = verbose

    def _trace(self, message):
        """
        Mencetak pesan jika mode verbose diaktifkan.
        """
        if self.verbose:
            print(message)


    def display_Lk(self, Lk, k):
        """
        Menampilkan frequent itemsets Lk dalam format tabel.

        Metode ini menampilkan itemsets sering muncul (frequent itemsets) dari kumpulan Lk 
        yang telah dihasilkan oleh algoritma Apriori. Informasi yang ditampilkan meliputi 
        itemsets dan nilai support mereka.

        Args:
            Lk (dict): Frequent itemsets pada level k.
            k (int): Level itemsets yang sedang ditampilkan.

        Output:
            Tabel yang menampilkan frequent itemsets dan support mereka.
        """
        if self.verbose:
            if k>1:
                self._trace(f"\nMemfilter C{k} yang memenuhi minimum count, menjadi L{k}")
            itemsets = ['{' + ', '.join(sorted(itemset)) + '}' for itemset in Lk.keys()]
            supports = list(Lk.values())
            df = pd.DataFrame({'Itemsets/Lk': itemsets, 'Support': supports})
            print(f"\nL{k} (Frequent Itemsets):\n{df.to_markdown(index=False)}")


    def display_Ck(self, Ck_counts, min_count, k):
        """
        Menampilkan kandidat itemsets (Ck) dan jumlah kemunculannya dalam format tabel 
        bersama dengan nilai hitungan minimum untuk memenuhi support minimum.

        Metode ini menampilkan itemsets kandidat dari kumpulan Ck yang dihasilkan oleh 
        algoritma Apriori bersama dengan jumlah kemunculan masing-masing itemset dalam 
        data transaksi. Juga menampilkan hitungan minimum yang diperlukan untuk itemset 
        agar dianggap sebagai frequent itemset.

        Args:
            Ck_counts (dict): Kamus yang memetakan kandidat itemsets ke jumlah kemunculannya.
            min_count (int): Jumlah minimum kemunculan untuk memenuhi support minimum.
            k (int): Level itemsets kandidat yang sedang ditampilkan.

        Output:
            Tabel yang menampilkan kandidat itemsets dan jumlah kemunculannya.
        """
        if self.verbose:
            print(f"\nMinimum count to meet min support (C{k}): {min_count}")
            candidates = ['{' + ', '.join(sorted(candidate)) + '}' for candidate in Ck_counts.keys()]
            counts = list(Ck_counts.values())
            df = pd.DataFrame({f'C{k}': candidates, 'Count': counts})
            print(f"\nC{k} (Candidate Itemsets) Counts:\n{df.to_markdown(index=False)}")

    def display_large_itemsets(self):
        """
        Menampilkan semua Large Itemset yang ditemukan selama proses Apriori.
        """
        if self.verbose:
            print("\nLarge Itemsets yang Ditemukan:")
            all_itemsets_info = [] 
            for k, Lk in enumerate(self.frequent_itemsets, start=1):
                for itemset, support in Lk.items():
                    itemset_str = '{' + ', '.join(sorted(itemset)) + '}'
                    all_itemsets_info.append({
                        'Lk': f'L{k}',
                        'Itemset': itemset_str,
                        'Support': support
                    })
            df_all_itemsets = pd.DataFrame(all_itemsets_info)
            print(df_all_itemsets.to_markdown(index=False))

    def fit(self, data):
        """
        Menjalankan algoritma Apriori pada dataset yang diberikan.
        Proses ini meliputi pembentukan frequent 1-itemsets, generasi kandidat itemsets,
        pemindaian transaksi untuk menentukan frequent itemsets, dan akhirnya,
        generasi aturan asosiasi dari itemsets yang ditemukan.

        Args:
            data (pd.DataFrame atau list of lists): Dataset transaksi untuk analisis.
        """
        transactions = self.process_input(data)
        itemset = sorted(set(itertools.chain.from_iterable(transactions)))
        
        L1 = self.create_L1(itemset, transactions)

        current_L = L1
        self.display_Lk(current_L, 1)  
        k = 2
        while current_L:
            self.frequent_itemsets.append(current_L)
            Ck = self.apriori_gen(current_L, k)
            Ck_list = list(Ck)
            if Ck_list:  
                current_L = self.scan_transactions(Ck_list, transactions, k)
                if current_L:
                    self.display_Lk(current_L, k)
                    k += 1
                else:                    
                    self._trace(f"\nTidak ada L{k} yang memenuhi min support, pencarian Large Itemset berhenti")
                    break
            else:  
                self._trace(f"\nC{k} kosong maka tidak mungkin ada L{k}, pencarian Large Itemset berhenti")
                break
        self.display_large_itemsets()  
        self.generate_rules(transactions)

    def process_input(self, data):
        """
        Memproses data input untuk mempersiapkan transaksi.

        Args:
            data (pd.DataFrame atau list of lists): Dataset yang berisi transaksi.

        Returns:
            list of sets: Daftar transaksi yang telah diproses.
        """
        self._trace("Memproses input data...")
        if isinstance(data, pd.DataFrame):
            transactions = data.apply(lambda x: set(x.dropna().astype(str)), axis=1).tolist()
        elif isinstance(data, list):
            transactions = [set(transaction) for transaction in data]
        return transactions

    def create_L1(self, itemset, transactions):
        """
        Membuat L1, kumpulan frequent 1-itemsets.

        Args:
            itemset (set): Set dari semua item unik dalam transaksi.
            transactions (list of sets): Daftar transaksi.

        Returns:
            dict: Frequent 1-itemsets dengan support mereka.
        """
        self._trace("Membuat L1 (frequent 1-itemsets)...")
        L1 = {}
        transaction_count = len(transactions)
        for item in itemset:
            item_count = sum(1 for transaction in transactions if item in transaction)
            support = item_count / transaction_count
            if support >= self.min_support:
                L1[frozenset([item])] = support
        return L1

    def add_text_effect(self, text, effect_type=None):
        """
        Menerapkan efek teks tertentu pada string.
        """
        effects = {
            'underline': '\033[4m',
            'strikethrough': '\033[9m',
            'reset': '\033[0m'
        }
        effect_code = effects.get(effect_type, '')
        reset_code = effects['reset'] if effect_type else ''
        return f"{effect_code}{text}{reset_code}"

    def underline_parts(self, sequence, start, end):
        """
        Menerapkan underline pada bagian tertentu dari sequence berdasarkan indeks.
        """
        return ', '.join(
            self.add_text_effect(item, 'underline') if start <= idx < end else item
            for idx, item in enumerate(sequence)
        )

    def apriori_gen(self, Lk_minus_1, k):
        """
        Menghasilkan kandidat k-itemsets (Ck) dari frequent (k-1)-itemsets (Lk-1).

        Args:
            Lk_minus_1 (dict): Frequent (k-1)-itemsets.
            k (int): Ukuran itemsets yang diinginkan.

        Returns:
            map object: Kandidat k-itemsets.
        """
        self._trace(f"\nMembuat C{k} (candidate k-itemsets)...")
        Ck = []
        if self.verbose:
            join_info = []  
            prune_info = []  
            Ck_info=[]
        Lk_minus_1 = list(Lk_minus_1.keys())

        for i in range(len(Lk_minus_1)):
            for j in range(i+1, len(Lk_minus_1)):
                l1, l2 = list(Lk_minus_1[i]), list(Lk_minus_1[j])
                l1.sort()
                l2.sort()
                
                l1_part = l1[1:k-1] if k > 2 else []
                l2_part = l2[0:k-2] if k > 2 else []
                if l1_part == l2_part:
                    new_candidate = l1[0:k-2] + [l1[k-2]] + [l2[k-2]]
                    all_subsets = list(itertools.combinations(new_candidate, k-1))
                    pruned = not all(frozenset(subset) in Lk_minus_1 for subset in all_subsets)
                    if not pruned:
                        Ck.append(new_candidate)
                        if self.verbose:
                            Ck_info.append({f'C{k}': '{' + ', '.join(new_candidate) + '}'})
                    if self.verbose:
                        join_info.append({
                            f'l{k-1}₁': '{' + self.underline_parts(l1, 1, k-1) + '}',
                            f'l{k-1}₂': '{' + self.underline_parts(l2, 0, k-2) + '}',
                            f'C{k}': '{' + ', '.join(new_candidate) + '}'
                        })
                        
                        subsets_str = [
                            '{' + (self.add_text_effect(', '.join(subset), 'strikethrough') 
                                if frozenset(subset) not in Lk_minus_1 else ', '.join(subset)) + '}'
                            for subset in all_subsets
                        ]
                        prune_info.append({
                            f'C{k}': '{' + ', '.join(new_candidate) + '}',
                            f'Subset-{k-1}': ', '.join(subsets_str),
                            'Pruned': 'Yes' if pruned else 'No'
                        })

        
        if self.verbose and join_info:
            print("\nTahap Join pada apriori-gen:")
            df_join = pd.DataFrame(join_info)
            print(df_join.to_markdown(index=False))

        
        if self.verbose and prune_info:
            print("\nTahap Prune pada apriori-gen:")
            df_prune = pd.DataFrame(prune_info)
            print(df_prune.to_markdown(index=False))

       
        if self.verbose:
            if Ck_info:
                print(f"\nC{k} final setelah prune:")
                df_Ck_final = pd.DataFrame(Ck_info)
                print(df_Ck_final.to_markdown(index=False))
            else:
                print(f"\nC{k} final setelah prune:")
                print(pd.DataFrame([{f'C{k}': '{}'}]).to_markdown(index=False))


        return map(frozenset, Ck)


    def scan_transactions(self, Ck, transactions, k):
        """
        Memindai transaksi untuk menghitung frekuensi munculnya setiap kandidat k-itemsets.

        Args:
            Ck (list): Kandidat k-itemsets.
            transactions (list of sets): Daftar transaksi.
            k (int): Ukuran k-itemsets.

        Returns:
            dict: Frequent k-itemsets dengan support mereka.
        """
        self._trace(f"\nMemindai transaksi untuk C{k}...")
        counts = defaultdict(int)
        Ck_list = list(Ck) 
        
        for transaction in transactions:
            transaction_set = set(transaction)
            for candidate in Ck_list:
                if candidate.issubset(transaction_set):
                    counts[candidate] += 1

        transaction_count = len(transactions)
        min_count = int(self.min_support * transaction_count + 0.5) 

        
        self.display_Ck(counts, min_count, k)

        return {item: count / transaction_count for item, count in counts.items() if count / transaction_count >= self.min_support}
    
    def generate_rules(self, transactions):
        """
        Menghasilkan aturan asosiasi dari frequent itemsets yang ditemukan.

        Args:
            transactions (list of sets): Daftar transaksi.
        """
        self._trace("\nMenghasilkan aturan asosiasi...")
        if self.verbose:
            rules_info = [] 
        for k, itemset in enumerate(self.frequent_itemsets[1:], start=2):  # Mulai dari L2
            for items, support in itemset.items():
                items_sorted = tuple(sorted(items))  
                for size in range(1, k): 
                    subsets = [tuple(sorted(x)) for x in itertools.combinations(items_sorted, size)]
                    for A in subsets:
                        A_set = frozenset(A)
                        B_set = items.difference(A_set)
                        B_sorted = tuple(sorted(B_set))  

                        if B_set:
                            AB_support = support
                            A_support = self.frequent_itemsets[len(A)-1].get(A_set, 0)
                            B_support = self.frequent_itemsets[len(B_set)-1].get(B_set, 0)

                            if A_support > 0 and B_support > 0:
                                confidence = AB_support / A_support
                                lift = confidence / B_support
                            else:
                                confidence = 0
                                lift = 0

                            if confidence >= self.min_confidence and lift >= self.min_lift:
                                rule = (A, B_sorted, AB_support, confidence, lift)
                                self.rules.append(rule)
                                if self.verbose:
                                    rules_info.append({
                                        'Antecedent (A)': '{'+', '.join(A) + '}',
                                        'Consequent (B)': '{'+', '.join(B_sorted) + '}',
                                        'Support': support,
                                        'Confidence': confidence,
                                        'Lift': lift
                                    })
        self._trace("")
        self._trace(f"Min. Support      : {self.min_support}")
        self._trace(f"Min. Confidence   : {self.min_confidence}")
        self._trace(f"Min. Lift         : {self.min_lift}")
        if self.verbose and rules_info:
            print(f"\nAturan Asosiasi yang Dihasilkan ({len(rules_info)}): ")
            df_rules = pd.DataFrame(rules_info)
            print(df_rules.to_markdown(index=False))



    def display_frequent_itemsets(self):
        """
        Menampilkan semua frequent itemsets yang telah ditemukan dalam bentuk DataFrame.
        """
        all_itemsets = []
        for k, Lk in enumerate(self.frequent_itemsets):
            for itemset, support in Lk.items():
                sorted_itemset = ', '.join(sorted(itemset))
                all_itemsets.append([sorted_itemset, support, len(itemset)])
        return pd.DataFrame(all_itemsets, columns=['Itemset', 'Support', 'Length'])

    def display_rules(self):
        """
        Menampilkan semua aturan asosiasi yang telah dihasilkan dalam bentuk DataFrame.
        """
        rules_df = pd.DataFrame(self.rules, columns=['Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift'])
        rules_df['Antecedent'] = rules_df['Antecedent'].apply(lambda x: ', '.join(x))
        rules_df['Consequent'] = rules_df['Consequent'].apply(lambda x: ', '.join(x))
        return rules_df
    