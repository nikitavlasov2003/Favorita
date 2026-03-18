class SimpleFavoritaLSTM(nn.Module):
    def __init__(self, dims, hidden_size=64, horizon=16, embed_dim=8):
        super().__init__()
        # Эмбеддинги (плюс 1 для запаса на неизвестные классы)
        self.s_emb = nn.Embedding(dims['n_store_nbrs'] + 1, embed_dim)
        self.i_emb = nn.Embedding(dims['n_item_nbrs'] + 1, embed_dim)
        self.f_emb = nn.Embedding(dims['n_familys'] + 1, embed_dim)
        self.c_emb = nn.Embedding(dims['n_clusters'] + 1, embed_dim)
        
        # LSTM (вход: 1 продажа + 5 экзогенных = 6)
        self.lstm = nn.LSTM(input_size=6, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size)
        
        # Декодер преобразует вектор в прогноз на 16 дней
        fc_in = hidden_size + (4 * embed_dim) + 1 + (5 * horizon)
        self.fc = nn.Sequential(
            nn.Linear(fc_in, 128),
            nn.ReLU(),
            nn.Linear(128, horizon)
        )

    def forward(self, batch):
        _, (h_n, _) = self.lstm(batch["x_enc"])
        h = self.bn(h_n[-1])
        
        s = self.s_emb(batch["store_id"])
        i = self.i_emb(batch["item_id"])
        f = self.f_emb(batch["family_id"])
        c = self.c_emb(batch["cluster_id"])
        
        x_dec_flat = batch["x_dec"].reshape(batch["x_dec"].size(0), -1)
        
        # Склеиваем всё вместе
        combined = torch.cat([h, s, i, f, c, batch["perishable"].unsqueeze(1), x_dec_flat], dim=1)
        return self.fc(combined)