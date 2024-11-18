const ValidationUtils = {
    validateEmail: (email) => {
      const re = /^(([^<>()[\]\\.,;:\s@"]+(\.[^<>()[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
      return re.test(String(email).toLowerCase());
    },

    validatePassword: (password) => {
      // パスワードの複雑性要件を実装
      // (例: 最小文字数, 英数字混在, 特殊文字を含むなど)
      return password.length >= 8;
    },

    // 他の検証用ユーティリティ関数
    // ...
  };

  export default ValidationUtils;