const DateUtils = {
    formatDate: (date) => {
      const options = { year: 'numeric', month: 'long', day: 'numeric' };
      return new Date(date).toLocaleDateString(undefined, options);
    },

    // 他の日付関連のユーティリティ関数
    // ...
  };

  export default DateUtils;