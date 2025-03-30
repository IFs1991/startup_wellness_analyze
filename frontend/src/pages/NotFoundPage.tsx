import { Link } from 'react-router-dom';

const NotFoundPage = () => {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4">
      <div className="text-center">
        <h1 className="text-9xl font-bold text-gray-800">404</h1>
        <h2 className="text-4xl font-bold text-gray-700 mt-4">ページが見つかりません</h2>
        <p className="text-gray-600 mt-4 mb-8">
          お探しのページは存在しないか、移動された可能性があります。
        </p>
        <Link
          to="/"
          className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
        >
          ホームに戻る
        </Link>
      </div>
    </div>
  );
};

export default NotFoundPage;